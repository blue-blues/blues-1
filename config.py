import torch
from dataclasses import dataclass
from typing import Optional
from setting import DEFAULT_SETTINGS
import os

@dataclass
class BluesConfig:
    # Model architecture for 1.2B-2B parameters
    n_layer: int = 32            # Increased from 6 to 32
    n_head: int = 32            # Increased from 8 to 32
    n_embd: int = 2048          # Increased from 512 to 2048
    head_dim: int = 64          # Keeping head_dim same
    vocab_size: int = 100300    # Same vocab size
    block_size: int = 2048      # Increased from 512 to 2048
    max_position_embeddings: int = 2048  # Match block_size
    bias: bool = False
    
    # Optimized MQA settings for large models
    num_key_value_heads: int = 8    # Increased KV heads
    num_key_value_groups: int = n_head // num_key_value_heads
    use_multiquery: bool = True
    
    # RMSNorm settings
    rms_norm_eps: float = 1e-5
    use_scale: bool = True
    
    # Enhanced MoE settings for better scaling
    tot_num_experts: int = 16       # Increased from 4 to 16
    chosen_num_experts: int = 2      # Using 2 experts per token
    embedding_multiplier_scale: int = 4  # Increased for better capacity
    noise_std: float = 0.1          # Reduced noise for stability
    lambadada: float = 0.005        # Reduced MoE loss coefficient
    
    # RoPE settings optimized for longer sequences
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = 1.0  # Enable RoPE scaling
    rope_scaling_factor: float = 1.0
    rope_ntk_flag: bool = True      # Enable NTK scaling
    use_dynamic_ntk: bool = True    # Enable dynamic NTK
    
    # Dynamic Expert Pruning settings - moved up with other primary settings
    enable_expert_pruning: bool = True
    pruning_interval: int = 1000    # Update pruning every N steps
    min_expert_capacity: float = 0.2 # Minimum capacity before pruning
    max_expert_growth: float = 1.5   # Maximum growth factor for experts
    pruning_threshold: float = 0.01  # Activity threshold for pruning
    growth_factor: float = 1.2       # Growth factor for active experts
    expert_weights_lr: float = 1e-3  # Learning rate for expert importance
    pruning_history_size: int = 100  # Size of history buffer for pruning decisions
    
    # Projection settings (must be defined before any optional settings)
    projection_dim: Optional[int] = None    # Will be set to n_embd if None
    use_projection: bool = True   # Enable projection layer by default
    
    # Special tokens with default values
    pad_token_id: int = 0          # Default pad token ID
    start_token_id: int = 1        # Default start token ID
    end_token_id: int = 2          # Default end token ID
    
    # Contrastive Learning settings
    temperature: float = 0.07        # Temperature for contrastive loss
    use_contrastive: bool = True     # Enable contrastive learning
    contrastive_loss_weight: float = 0.1  # Weight for contrastive loss (renamed from contrastive_weight)
    projection_size: int = 128       # Size of projection head output
    queue_size: int = 65536         # Memory queue size for contrastive learning
    momentum: float = 0.999         # Momentum encoder update rate
    
    # Alias attributes for model compatibility
    @property
    def hidden_size(self):
        return self.n_embd
        
    @property
    def num_attention_heads(self):
        return self.n_head
        
    @property
    def num_hidden_layers(self):
        return self.n_layer
    
    @property
    def num_layers(self):  # Alias for compatibility
        return self.n_layer
    
    # Update property getters for RoPE compatibility
    @property
    def rotary_emb_base(self):
        return self.rope_theta
    
    @property
    def rope_scaling_type(self):
        return "linear" if self.rope_scaling else "none"
    
    # Device setting
    device: str = DEFAULT_SETTINGS['device']
    
    # Training stability settings
    dropout: float = 0.1            # Increased dropout
    gradient_checkpointing: bool = True  # Enable by default
    
    # Expert settings
    num_experts: int = 16           # Match tot_num_experts
    expert_capacity: int = 64       # Increased capacity
    moe_layers: list = None
    expert_ffn_size: int = None     # Will be 4 * hidden_size in post_init
    top_k: int = 2                  # Keep top 2 experts
    use_moe: bool = True            # Enable MoE by default
    
    # Model parallel settings
    expert_parallel: bool = True
    tensor_parallel: bool = True
    pipeline_parallel: bool = False
    
    # Flash attention settings
    flash_attn: bool = DEFAULT_SETTINGS['flash_optimizations']['use_flash_attn']
    mem_efficient: bool = True
    
    # Quantization
    bits: Optional[int] = None
    
    # Updated MoE configuration
    use_moe: bool = True
    
    def __post_init__(self):
        # Handle projection settings first
        if self.projection_dim is None:
            self.projection_dim = self.n_embd
            
        # Validate projection settings
        if self.use_projection:
            assert self.projection_dim > 0, "projection_dim must be positive"
            
        # Verify and adjust dimensions
        assert self.n_embd % self.n_head == 0, "Embedding dim must be divisible by num heads"
        self.head_dim = self.n_embd // self.n_head
        
        # Verify MQA settings
        assert self.n_head % self.num_key_value_heads == 0, \
            "Number of attention heads must be divisible by number of key/value heads"
        assert self.num_key_value_heads <= self.n_head, \
            "Number of key/value heads must be <= number of attention heads"
        
        # Setup MoE layers
        if self.moe_layers is None:
            self.moe_layers = list(range(1, self.n_layer, 2))
        
        # Verify MoE settings
        assert self.chosen_num_experts <= self.tot_num_experts, \
            "Chosen experts must be less than or equal to total experts"
        
        # Set expert FFN size if not explicitly specified
        if self.expert_ffn_size is None:
            self.expert_ffn_size = 4 * self.hidden_size
            
        # Verify MoE settings
        assert self.top_k <= self.num_experts, \
            "top_k must be less than or equal to num_experts"
        
        # Additional validations for large model
        assert self.block_size == self.max_position_embeddings, \
            "block_size must match max_position_embeddings"
        assert self.n_embd % self.n_head == 0, \
            "embedding dimension must be divisible by number of heads"
        
        # Validate pruning settings
        if self.enable_expert_pruning:
            assert 0 < self.min_expert_capacity < 1, "min_expert_capacity must be between 0 and 1"
            assert self.max_expert_growth > 1, "max_expert_growth must be greater than 1"
            assert 0 < self.pruning_threshold < 1, "pruning_threshold must be between 0 and 1"
            assert self.pruning_interval > 0, "pruning_interval must be positive"
            assert self.pruning_history_size > 0, "pruning_history_size must be positive"
        
        # Validate contrastive learning settings
        if self.use_contrastive:
            assert self.temperature > 0, "temperature must be positive"
            assert 0 <= self.contrastive_loss_weight <= 1, "contrastive_loss_weight must be between 0 and 1"
            assert self.projection_size > 0, "projection_size must be positive"
            assert self.queue_size > 0, "queue_size must be positive"
            assert 0 <= self.momentum <= 1, "momentum must be between 0 and 1"
        
        # Set projection dimension if not specified
        if self.projection_dim is None:
            self.projection_dim = self.n_embd
        
        # Validate projection settings
        if self.use_projection:
            assert self.projection_dim > 0, "projection_dim must be positive"
        
        # Verify special tokens are properly set
        assert isinstance(self.pad_token_id, int), "pad_token_id must be an integer"
        assert isinstance(self.start_token_id, int), "start_token_id must be an integer"
        assert isinstance(self.end_token_id, int), "end_token_id must be an integer"
        assert self.pad_token_id >= 0, "pad_token_id must be non-negative"
        
        # Calculate and print model size
        num_params = self._calculate_params()
        print(f"Model size: {num_params/1e9:.2f}B parameters")
    
    def _calculate_params(self):
        """Calculate approximate number of parameters"""
        # Embeddings
        embed_params = self.vocab_size * self.n_embd
        pos_embed_params = self.max_position_embeddings * self.n_embd
        
        # Per layer
        mha_params = 4 * self.n_embd * self.n_embd  # Q,K,V,O matrices
        expert_params = self.num_experts * (
            2 * self.n_embd * self.expert_ffn_size +  # Up/down projections
            self.expert_ffn_size * self.n_embd        # Output projection
        )
        
        # Total
        total_params = (
            embed_params +
            pos_embed_params +
            (mha_params + expert_params) * self.n_layer
        )
        
        return total_params

    def set_special_tokens(self, pad_token, start_token, end_token):
        """Set special token IDs after tokenizer is initialized"""
        self.pad_token_id = int(pad_token) if pad_token is not None else 0
        self.start_token_id = int(start_token) if start_token is not None else 1
        self.end_token_id = int(end_token) if end_token is not None else 2

    def update_gpu_settings(self, flash_attn_available: bool, deepspeed_available: bool):
        """Update settings based on GPU feature availability"""
        # Only enable flash attention if GPU is compatible
        self.flash_attn = flash_attn_available
        self.mem_efficient = flash_attn_available
        
        if not flash_attn_available:
            print("Using standard attention mechanism")
        
        # Update parallel settings based on DeepSpeed availability
        self.expert_parallel = deepspeed_available
        self.tensor_parallel = deepspeed_available
        
        # Adjust model settings for standard attention if needed
        if not flash_attn_available:
            # Use more memory-efficient settings for standard attention
            self.gradient_checkpointing = True
            self.num_key_value_heads = min(self.num_key_value_heads, 4)
            print("Adjusted model settings for standard attention")
    
    def update_moe_settings(self, num_experts=None, top_k=None):
        """Update MoE settings based on available resources"""
        if num_experts is not None:
            self.num_experts = num_experts
        if top_k is not None:
            self.top_k = min(top_k, self.num_experts)

# Add checkpoint directory configuration
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
data_cache_dir = os.path.join(os.path.dirname(__file__), 'data_cache')

config = BluesConfig()
