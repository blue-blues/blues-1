import torch
import os
vocab_size = 50000           # Keeping vocabulary size the same
n_embed = 1024               # Embedding size increased to 1024
n_head = 16                  # Number of attention heads increased to 16
num_experts = 32             # Increased number of experts to 32 (if using MoE)
top_k = 4                    # Keeping top_k the same
n_layer = 12                 # Increased number of layers to 12
block_size = 512             # Keeping block_size the same
dropout = 0.1                # Keeping dropout the same
learning_rate = 1e-4         # Learning rate remains unchanged
eval_iters = 10              # Keeping eval_iters the same
batch_size = 32              # Keeping batch_size the same

# Device settings
device_type = "cuda"  # or "cpu"
device = torch.device(device_type if torch.cuda.is_available() else "cpu")

# Dataset settings
dataset = "test1"
data_dir = os.path.join("data", dataset)