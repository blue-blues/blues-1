import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

class ExpertMetrics:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset_metrics()
    
    def reset_metrics(self):
        self.usage_counts = torch.zeros(self.num_experts)
        self.importance_scores = torch.ones(self.num_experts)
        self.specialization_scores = torch.zeros(self.num_experts)
        self.load_balance_history = []
        
    def update_metrics(self, routing_probs: torch.Tensor, expert_outputs: torch.Tensor):
        # Update usage counts
        self.usage_counts += routing_probs.sum(dim=(0, 1))
        
        # Calculate specialization scores using output variance
        mean_outputs = expert_outputs.mean(dim=1, keepdim=True)
        variance = ((expert_outputs - mean_outputs) ** 2).mean(dim=1)
        self.specialization_scores = (0.9 * self.specialization_scores + 0.1 * variance)
        
        # Calculate load balancing score
        total_tokens = routing_probs.sum()
        expert_loads = routing_probs.sum(dim=(0, 1)) / total_tokens
        load_balance = (expert_loads.max() / (expert_loads.mean() + 1e-5)).item()
        self.load_balance_history.append(load_balance)

class AdaptiveRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int = 2, capacity_factor: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, num_experts)
        )
        
        # Adaptive routing threshold
        self.register_buffer('routing_thresholds', torch.ones(num_experts) * 0.1)
        
        # Expert pruning threshold
        self.min_usage_threshold = 0.01
        self.metrics = ExpertMetrics(num_experts)
        
        # Temperature annealing
        self.initial_temperature = 1.0
        self.min_temperature = 0.1
        self.temperature = self.initial_temperature
        
    def anneal_temperature(self, step: int, total_steps: int):
        """Anneal routing temperature over training"""
        self.temperature = max(
            self.min_temperature,
            self.initial_temperature * (1 - step / total_steps)
        )
    
    def update_thresholds(self, expert_metrics: ExpertMetrics):
        """Adaptively update routing thresholds based on expert performance"""
        usage_ratios = expert_metrics.usage_counts / expert_metrics.usage_counts.sum()
        specialization = F.softmax(expert_metrics.specialization_scores, dim=0)
        
        # Increase thresholds for overused or underspecialized experts
        self.routing_thresholds = 0.1 + 0.2 * (usage_ratios + (1 - specialization))
    
    def prune_experts(self) -> torch.Tensor:
        """Return mask for active experts based on usage and specialization"""
        usage_ratios = self.metrics.usage_counts / self.metrics.usage_counts.sum()
        specialization = F.softmax(self.metrics.specialization_scores, dim=0)
        
        # Experts are kept if they have sufficient usage and specialization
        keep_mask = (usage_ratios > self.min_usage_threshold) & (specialization > 0.05)
        return keep_mask
    
    def forward(self, inputs: torch.Tensor, step: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = inputs.shape
        
        # Get router logits
        router_logits = self.router(inputs)
        
        # Apply temperature scaling
        router_logits = router_logits / self.temperature
        
        # Get expert mask for pruning
        expert_mask = self.prune_experts()
        router_logits = router_logits.masked_fill(~expert_mask, float('-inf'))
        
        # Calculate routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Apply routing thresholds
        routing_probs = routing_probs * (routing_probs > self.routing_thresholds)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize probabilities
        top_k_probs_sum = top_k_probs.sum(dim=-1, keepdim=True)
        top_k_probs = top_k_probs / (top_k_probs_sum + 1e-9)
        
        # Create routing weights tensor
        routing_weights = torch.zeros_like(routing_probs)
        routing_weights.scatter_(-1, top_k_indices, top_k_probs)
        
        return routing_weights, expert_mask

class EnhancedMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.expert_size = config.expert_size
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, self.expert_size),
                nn.GELU(),
                nn.Linear(self.expert_size, self.input_size)
            ) for _ in range(self.num_experts)
        ])
        
        # Enhanced router with adaptive thresholds
        self.router = AdaptiveRouter(
            input_size=self.input_size,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        
        # Expert dropout
        self.expert_dropout = nn.Dropout(config.expert_dropout)
        
        # Metrics tracking
        self.metrics = self.router.metrics
    
    def forward(self, inputs: torch.Tensor, step: int = None) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, hidden_size = inputs.shape
        
        # Get routing weights and expert mask
        routing_weights, expert_mask = self.router(inputs, step)
        
        # Process inputs through experts
        expert_outputs = torch.zeros(
            self.num_experts, batch_size * seq_len, hidden_size,
            device=inputs.device
        )
        
        flat_inputs = inputs.view(-1, hidden_size)
        for i, expert in enumerate(self.experts):
            if expert_mask[i]:
                expert_outputs[i] = expert(flat_inputs)
        
        # Apply expert dropout
        expert_outputs = self.expert_dropout(expert_outputs)
        
        # Combine expert outputs
        combined_outputs = torch.einsum(
            'be,ebh->bh',
            routing_weights.view(-1, self.num_experts),
            expert_outputs
        )
        
        # Update metrics
        self.metrics.update_metrics(routing_weights, expert_outputs)
        
        # Reshape output
        outputs = combined_outputs.view(batch_size, seq_len, hidden_size)
        
        # Return outputs and metrics
        metrics = {
            'routing_weights': routing_weights,
            'expert_mask': expert_mask,
            'load_balance': self.metrics.load_balance_history[-1],
            'specialization': self.metrics.specialization_scores
        }
        
        return outputs, metrics
