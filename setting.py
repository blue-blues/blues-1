# Training settings
learning_rate = 3e-4
dropout = 0.2
batch_size = 128
weight_decay = 0.02
max_iters = 20000
eval_interval = 2000

# Data settings
dataset = "df_file.csv"
data_cache_dir = "data_cache"  # Directory to store processed data
chunk_size = 1000  # Number of rows per chunk
chunk_memory_limit = 1024 * 1024 * 512  # 512MB per chunk
verify_data_loading = True  # Add verification flag
checkpoint_dir = "checkpoints"

# Generation settings
generation_config = {
    'temperature': 0.8,
    'top_p': 0.9,
    'top_k': 50,
    'max_length': 100
}

# Performance settings
mixed_precision = True
gradient_accumulation_steps = 4
warmup_steps = 100
lr_decay_steps = 1000
min_lr = 1e-5

# Memory optimization
pin_memory = True
num_workers = 4
prefetch_factor = 2

# Added optimization settings
optimizer_config = {
    'betas': (0.9, 0.95),
    'eps': 1e-8,
    'weight_decay': 0.02
}

# Flash Attention settings
flash_attention_config = {
    'max_seqlen': 2048,
    'dropout_p': 0.1,
    'causal': True,
    'return_attn_probs': False,
    'deterministic': False
}

# Memory and optimization for Flash Attention
flash_optimizations = {
    'use_flash_attn': True,
    'mem_efficient': True,
    'enable_tiling': True,
    'tile_size': 256
}
