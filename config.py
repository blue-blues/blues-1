import torch
from dataclasses import dataclass
from typing import Optional
from setting import DEFAULT_SETTINGS
import os

@dataclass
class BluesConfig:
    # Reduced model architecture
    n_layer: int = 6           # Reduced from 12
    n_head: int = 8           # Reduced from 12
    n_embd: int = 512         # Reduced from 768
    head_dim: int = 64        # n_embd // n_head
    vocab_size: int = 200020
    block_size: int = 512     # Reduced from 1024
    max_position_embeddings: int = 512
    bias: bool = False
    
    # Reduced MQA settings
    num_key_value_heads: int = 2  # Reduced from 4
    num_key_value_groups: int = n_head // num_key_value_heads
    use_multiquery: bool = True
    
    # RMSNorm settings
    rms_norm_eps: float = 1e-5
    use_scale: bool = True
    
    # MoE settings (reduced)
    tot_num_experts: int = 4   # Reduced from 8
    chosen_num_experts: int = 1  # Reduced from 2
    embedding_multiplier_scale: int = 2  # Reduced from 4
    noise_std: float = 1.0  # Noise for expert routing
    lambadada: float = 0.01  # MoE loss coefficient
    
    # RoPE settings
    rope_theta: float = 10000.0  # Base frequency
    rope_scaling: Optional[float] = None  # Scaling factor for RoPE
    rope_scaling_factor: float = 1.0
    rope_ntk_flag: bool = False  # NTK scaling flag
    use_dynamic_ntk: bool = False  # Dynamic NTK scaling
    
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
    
    # Training
    dropout: float = DEFAULT_SETTINGS['dropout']
    gradient_checkpointing: bool = False
    
    # Expert settings
    num_experts: int = 8
    expert_capacity: int = 32
    moe_layers: list = None
    
    # Model parallel settings
    expert_parallel: bool = False
    tensor_parallel: bool = False
    pipeline_parallel: bool = False
    
    # Flash attention settings
    flash_attn: bool = DEFAULT_SETTINGS['flash_optimizations']['use_flash_attn']
    mem_efficient: bool = DEFAULT_SETTINGS['flash_optimizations']['mem_efficient']
    
    # Special tokens (will be set later)
    pad_token_id: Optional[int] = None
    start_token_id: Optional[int] = None
    end_token_id: Optional[int] = None
    
    # Quantization
    bits: Optional[int] = None
    
    def __post_init__(self):
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

    def set_special_tokens(self, pad_token, start_token, end_token):
        """Set special token IDs after tokenizer is initialized"""
        self.pad_token_id = pad_token
        self.start_token_id = start_token
        self.end_token_id = end_token

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

# Add checkpoint directory configuration
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
data_cache_dir = os.path.join(os.path.dirname(__file__), 'data_cache')

config = BluesConfig()
