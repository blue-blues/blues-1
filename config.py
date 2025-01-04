import dataclasses
import torch

from model import *


@dataclasses.dataclass
class config:
    # Model architecture
    vocab_size = 200019  # Updated to match tiktoken vocabulary size
    max_position_embeddings = 256
    num_layers = 4
    hidden_size = 192
    head_dim = 48

    # Attention configuration
    num_attention_heads = 4
    num_key_value_heads = 2
    
    # MoE configuration
    embedding_multiplier_scale = 4
    tot_num_experts = 4
    chosen_num_experts = 1
    noise_std = 0.05
    lambadada = 0.5

    # Layer normalization
    rms_norm_eps = 1e-5
    use_scale = True
    rope_theta = 10000.0

    # Training
    dropout = 0.1

    # Hardware optimization flags
    use_flash_attn = True  # Will be automatically disabled if not available
    use_deepspeed = True  # Will be automatically disabled if not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
