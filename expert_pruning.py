import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
import numpy as np
from dataclasses import dataclass
from collections import deque

@dataclass
class PruningState:
    """Track pruning state and metrics"""
    active_experts: List[int]
    usage_history: Dict[int, deque]
    pruned_experts: List[int]
    step: int
    last_prune_step: int

class DynamicExpertPruning:
    def __init__(self, config, history_size: int = 100):
        self.config = config
        self.history_size = history_size
        self.state = PruningState(
            active_experts=list(range(config.num_experts)),
            usage_history={i: deque(maxlen=history_size) for i in range(config.num_experts)},
            pruned_experts=[],
            step=0,
            last_prune_step=0
        )
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for pruning events"""
        self.logger = logging.getLogger('expert_pruning')
        self.logger.setLevel(logging.INFO)
        
    def update(self, expert_ids: torch.Tensor, router_probs: torch.Tensor) -> None:
        """Update expert usage statistics"""
        self.state.step += 1
        batch_usage = self._compute_batch_usage(expert_ids, router_probs)
        self._update_history(batch_usage)
        
        if self._should_prune():
            self._perform_pruning_step()

    def _compute_batch_usage(self, expert_ids: torch.Tensor, router_probs: torch.Tensor) -> Dict[int, float]:
        """Compute expert usage for current batch"""
        usage = {}
        flat_ids = expert_ids.flatten()
        flat_probs = router_probs.flatten()
        
        unique_ids, counts = flat_ids.unique(return_counts=True)
        for idx, count in zip(unique_ids.tolist(), counts.tolist()):
            usage[idx] = count / len(flat_ids)
        return usage

    def _update_history(self, batch_usage: Dict[int, float]) -> None:
        """Update rolling usage history for each expert"""
        for expert_id in self.state.active_experts:
            usage = batch_usage.get(expert_id, 0.0)
            self.state.usage_history[expert_id].append(usage)

    def _should_prune(self) -> bool:
        """Determine if pruning should occur"""
        return (self.config.enable_expert_pruning and 
                self.state.step >= self.state.last_prune_step + self.config.pruning_interval)

    def _perform_pruning_step(self) -> None:
        """Execute pruning and growth operations"""
        mean_usage = self._compute_mean_usage()
        pruning_decisions = self._make_pruning_decisions(mean_usage)
        
        if pruning_decisions.get('to_prune'):
            self._apply_pruning(pruning_decisions['to_prune'])
            self._apply_growth(pruning_decisions['to_grow'])
            
        self.state.last_prune_step = self.state.step
        self._log_pruning_event(pruning_decisions)

    def _compute_mean_usage(self) -> Dict[int, float]:
        """Compute mean usage for each expert"""
        return {
            expert_id: np.mean(list(history)) 
            for expert_id, history in self.state.usage_history.items()
            if expert_id in self.state.active_experts
        }

    def _make_pruning_decisions(self, mean_usage: Dict[int, float]) -> Dict[str, List[int]]:
        """Determine which experts to prune and grow"""
        global_mean = np.mean(list(mean_usage.values()))
        to_prune = []
        to_grow = []
        
        for expert_id, usage in mean_usage.items():
            if usage < global_mean * self.config.pruning_threshold:
                to_prune.append(expert_id)
            elif usage > global_mean:
                to_grow.append(expert_id)
                
        return {'to_prune': to_prune, 'to_grow': to_grow}

    def apply_to_parameters(self, experts: nn.ModuleList) -> None:
        """Apply pruning and growth to expert parameters"""
        if not hasattr(self, '_last_update'):
            self._last_update = {}
            return

        for expert_id, expert in enumerate(experts):
            if expert_id in self.state.pruned_experts:
                self._scale_parameters(expert, self.config.min_expert_capacity)
            elif expert_id in self._last_update:
                self._scale_parameters(expert, self._last_update[expert_id])

    def _scale_parameters(self, expert: nn.Module, scale_factor: float) -> None:
        """Scale expert parameters by given factor"""
        with torch.no_grad():
            for param in expert.parameters():
                param.data *= scale_factor

    def _log_pruning_event(self, decisions: Dict[str, List[int]]) -> None:
        """Log pruning decisions and metrics"""
        self.logger.info(
            f"Step {self.state.step} - Pruned: {decisions['to_prune']}, "
            f"Grown: {decisions['to_grow']}"
        )

    def get_state_dict(self) -> Dict:
        """Get pruning state for checkpointing"""
        return {
            'step': self.state.step,
            'active_experts': self.state.active_experts,
            'pruned_experts': self.state.pruned_experts,
            'last_prune_step': self.state.last_prune_step,
            'usage_history': {k: list(v) for k, v in self.state.usage_history.items()}
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load pruning state from checkpoint"""
        self.state.step = state_dict['step']
        self.state.active_experts = state_dict['active_experts']
        self.state.pruned_experts = state_dict['pruned_experts']
        self.state.last_prune_step = state_dict['last_prune_step']
        self.state.usage_history = {
            k: deque(v, maxlen=self.history_size) 
            for k, v in state_dict['usage_history'].items()
        }

    def _apply_pruning(self, experts_to_prune: List[int]) -> None:
        """Apply pruning to selected experts"""
        for expert_id in experts_to_prune:
            if expert_id in self.state.active_experts:
                self.state.active_experts.remove(expert_id)
                self.state.pruned_experts.append(expert_id)
                self._last_update[expert_id] = self.config.min_expert_capacity
                self.logger.info(f"Pruned expert {expert_id}")

    def _apply_growth(self, experts_to_grow: List[int]) -> None:
        """Apply growth to selected experts"""
        if not experts_to_grow:
            return
            
        mean_usage = np.mean([
            np.mean(list(self.state.usage_history[i]))
            for i in experts_to_grow
        ])
        
        for expert_id in experts_to_grow:
            expert_usage = np.mean(list(self.state.usage_history[expert_id]))
            growth_factor = min(
                self.config.growth_factor * (expert_usage / mean_usage),
                self.config.max_expert_growth
            )
            self._last_update[expert_id] = growth_factor
            self.logger.info(f"Growing expert {expert_id} by factor {growth_factor:.2f}")

    def get_expert_stats(self) -> Dict[str, Dict]:
        """Get current expert statistics"""
        stats = {
            'active': {
                expert_id: np.mean(list(self.state.usage_history[expert_id]))
                for expert_id in self.state.active_experts
            },
            'pruned': {
                expert_id: np.mean(list(self.state.usage_history[expert_id]))
                for expert_id in self.state.pruned_experts
            },
            'total_steps': self.state.step,
            'last_prune': self.state.last_prune_step
        }
        return stats
