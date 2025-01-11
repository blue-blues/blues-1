import argparse
import json
import os
import torch
from pathlib import Path

# Define all default settings in a single dictionary
DEFAULT_SETTINGS = {
    # System settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'distributed': False,
    'local_rank': -1,

    # Training settings
    'learning_rate': 3e-4,
    'dropout': 0.1,
    'batch_size': 4,         # Increased from 1 due to smaller model
    'weight_decay': 0.02,
    'max_iters': 50000,
    'eval_interval': 500,
    'save_interval': 1000,
    'warmup_steps': 1000,
    'lr_decay_steps': 1000,
    'min_lr': 3e-5,
    'gradient_accumulation_steps': 32,  # Reduced from 64
    'max_grad_norm': 1.0,
    'min_batch_size': 1,

    # Data settings
    'dataset': "df_file.csv",
    'data_cache_dir': "data_cache",
    'chunk_size': 10000,        # Reduced from 1000
    'chunk_memory_limit': 256 * 1024 * 1024,  # 256MB per chunk
    'verify_data_loading': True,
    'checkpoint_dir': "checkpoints",
    'data_source': 'csv',  # Options: 'csv', 'huggingface'

    # HuggingFace dataset settings
    'hf_dataset_config': {
        'name': 'ytzi/the-stack-dedup-python-scored',
        'subset': None,
        'split': 'train',
        'streaming': True,
        'text_column': 'content',
        'cache_dir': 'hf_cache',
    },

    # Dataset merging settings
    'merged_data_dir': 'data_cache/merged',
    'dataset_weights': None,
    'datasets': {
        'paths': [],  # List of paths to different dataset directories
        'weights': [], # Optional weights for each dataset
        'merge_strategy': 'weighted',  # Options: 'weighted', 'equal', 'proportional'
    },

    # Tokenizer settings
    'tokenizer_config': {
        'name': 'o200k_base',
        'provider': 'tiktoken',
        'special_tokens': {
            'start_token': '<|start|>',
            'end_token': '<|end|>',
            'pad_token': '<|pad|>',
        },
        'max_seq_length': 512,  # Reduced from 2048
        'add_special_tokens': True,
    },

    # Flash optimizations
    'flash_optimizations': {
        'use_flash_attn': True,
        'mem_efficient': True,
        'enable_tiling': True,
        'tile_size': 128,     # Increased from 64
        'use_cuda_fp16': True,
    },

    # Optimizer settings
    'optimizer_config': {
        'betas': (0.9, 0.95),
        'eps': 1e-8,
        'weight_decay': 0.1,
        # Remove capturable setting as it's not supported on CPU
    },

    # Memory management
    'memory_limit_mb': 8000,  # Reduced memory limit
}

# Initialize all settings as module-level variables
globals().update(DEFAULT_SETTINGS)

# Additional settings initialization from DEFAULT_SETTINGS subsections
for section in ['optimizer_config', 'flash_optimizations', 'hf_dataset_config']:
    if section in DEFAULT_SETTINGS:
        globals()[section] = DEFAULT_SETTINGS[section]

# Extract dataset-related settings for easy access
dataset = DEFAULT_SETTINGS['dataset']
data_cache_dir = DEFAULT_SETTINGS['data_cache_dir']
chunk_size = DEFAULT_SETTINGS['chunk_size']

flash_optimizations = DEFAULT_SETTINGS['flash_optimizations']
optimizer_config = DEFAULT_SETTINGS.get('optimizer_config', {})

# Make gradient_accumulation_steps available globally
gradient_accumulation_steps = DEFAULT_SETTINGS['gradient_accumulation_steps']
warmup_steps = DEFAULT_SETTINGS['warmup_steps']
max_grad_norm = DEFAULT_SETTINGS['max_grad_norm']
min_batch_size = DEFAULT_SETTINGS['min_batch_size']

# Memory management
memory_limit_mb = 24000  # 8GB limit
min_batch_size = 1
max_batch_size = 32
