# Blues-01: MoE Language Model

A PyTorch implementation of a Mixture of Experts (MoE) language model with multi-query attention and efficient memory management.

## Features

- Mixture of Experts (MoE) architecture with dynamic routing
- Multi-Query Attention (MQA) with configurable key-value heads
- Flash Attention integration for GPU optimization
- Memory-efficient training with gradient checkpointing
- Load balancing for expert utilization
- Automatic mixed precision training
- DeepSpeed integration for distributed training
- Dynamic batch sizing based on available memory

## Model Architecture

### Core Components

1. **Multi-Query Attention (MQA)**
   - Configurable number of key-value heads
   - Optional Flash Attention support
   - Rotary position embeddings (RoPE)

2. **Mixture of Experts (MoE)**
   - Dynamic expert routing with noise
   - Load balancing through auxiliary loss
   - Configurable number of experts and routing

### Default Configuration

```python
config = BluesConfig(
    n_layer=6,                # Number of transformer layers
    n_head=8,                # Number of attention heads
    n_embd=512,             # Embedding dimension
    num_key_value_heads=2,   # Number of KV heads
    num_experts=8,           # Number of experts per MoE layer
    top_k=2,                # Number of experts to route to
    vocab_size=200020,      # Vocabulary size
    block_size=512          # Maximum sequence length
)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blues-moe.git
cd blues-moe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- flash-attn (optional, for GPU acceleration)
- deepspeed (optional, for distributed training)
- einops
- triton
- tiktoken

## Usage

### Training

1. Prepare your dataset:
```bash
python data.py --input_dir data/raw --cache_dir data_cache
```

2. Start training:
```bash
# Single GPU training
python train.py

# Multi-GPU training with DeepSpeed
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### Generation

```python
from model import blues
from config import config

# Load model
model = blues(config, tokenizer)
model.load_state_dict(torch.load('checkpoints/model.pt'))

# Generate text
output = model.generate(
    prompt="Once upon a time",
    output_len=100,
    temperature=0.8,
    top_p=0.9,
    top_k=50
)
```

## Memory Optimization

The implementation includes several memory optimization features:

1. **Gradient Checkpointing**
   - Enable with `model.gradient_checkpointing_enable()`
   - Trades computation for memory efficiency

2. **Dynamic Batch Sizing**
   - Automatically adjusts batch size based on available memory
   - Supports gradient accumulation

3. **Expert Pruning**
   - Removes unused experts during training
   - Balances expert utilization

4. **Memory-Efficient Attention**
   - Uses Flash Attention when available
   - Fallback to optimized standard attention

## Training Tips

1. **Starting Training**
   - Begin with a small number of experts (4-8)
   - Monitor expert utilization through the load balancing loss
   - Gradually increase model size based on performance

2. **Optimization Settings**
   - Adjust learning rate based on model size
   - Use gradient accumulation for larger batches
   - Enable gradient checkpointing for large models

3. **Memory Management**
   - Monitor GPU memory usage
   - Adjust batch size dynamically
   - Use mixed precision training

4. **Expert Configuration**
   - Start with top_k=2 for routing
   - Adjust number of experts based on dataset size
   - Monitor expert utilization metrics

## API Reference

### Configuration

```python
BluesConfig(
    n_layer: int = 6,
    n_head: int = 8,
    n_embd: int = 512,
    num_experts: int = 8,
    top_k: int = 2,
    expert_ffn_size: int = None,
    dropout: float = 0.1,
    flash_attn: bool = True
)
```

### Model Methods

```python
model.forward(input_ids, target_ids=None)
model.generate(prompt, output_len=100, temperature=0.8)
model.gradient_checkpointing_enable()
```

## Checkpointing

The model supports various checkpointing features:

```python
# Save checkpoint
save_checkpoint(model, optimizer, iter_num, loss, 'checkpoint.pt')

# Resume training
model, optimizer, start_iter = setup_training(resume_from='checkpoint.pt')
```

## TODO & Planned Improvements

### 1. Better Representation Learning
- [ ] Implement contrastive learning for improved embeddings
- [ ] Add supervised contrastive loss
- [ ] Support for multiple positive examples
- [ ] Temperature-scaled InfoNCE loss
- [ ] Hard negative mining strategies

### 2. Specialized Processing
- [x] Basic Mixture of Experts implementation
- [ ] Dynamic expert pruning
- [ ] Conditional computation paths
- [ ] Expert specialization metrics
- [ ] Adaptive routing strategies
- [ ] Expert load balancing improvements

### 3. Position Understanding
- [x] Basic RoPE implementation
- [ ] Dynamic RoPE scaling
- [ ] Improved position interpolation
- [ ] NTK-aware scaling
- [ ] Position aliasing reduction
- [ ] Extended context length support

### 4. Long Sequence Handling
- [ ] Sliding window attention
- [ ] Sparse attention patterns
- [ ] Local-global attention mixing
- [ ] Adaptive context window
- [ ] Memory-efficient sequence processing
- [ ] Progressive sequence chunking

### 5. Attention Optimization
- [x] Basic Flash Attention integration
- [ ] Flash Attention V2 support
- [ ] Automatic precision switching
- [ ] Custom CUDA kernels
- [ ] Memory access optimization
- [ ] Attention pattern pruning

### 6. Memory Efficient Inference
- [ ] KV cache implementation
- [ ] Quantized cache storage
- [ ] Dynamic cache pruning
- [ ] Prefetch optimization
- [ ] Cache compression strategies
- [ ] Memory-mapped cache support

### Priority Matrix

| Feature | Impact | Difficulty | Priority |
|---------|---------|------------|-----------|
| Contrastive Learning | High | Medium | 1 |
| Expert Pruning | High | Medium | 2 |
| Flash Attention V2 | High | Low | 3 |
| KV Cache | High | Medium | 4 |
| Sliding Window | Medium | High | 5 |
| Dynamic RoPE | Medium | Low | 6 |

### Implementation Notes

1. **Contrastive Learning**
   - Start with simple pairwise contrastive loss
   - Gradually add multi-positive support
   - Implement temperature scaling
   - Add data augmentation strategies

2. **Expert Improvements**
   - Monitor expert utilization
   - Implement adaptive routing
   - Add expert dropout
   - Optimize load balancing

3. **Position Enhancements**
   - Test different RoPE configurations
   - Benchmark position understanding
   - Optimize for longer sequences
   - Add position caching

4. **Sequence Processing**
   - Start with basic sliding window
   - Add attention sparsity
   - Implement efficient chunking
   - Optimize memory usage

5. **Attention Updates**
   - Upgrade to Flash Attention V2
   - Profile and optimize kernels
   - Add precision autotuning
   - Implement pattern caching

6. **Inference Optimization**
   - Design efficient KV cache
   - Add compression support
   - Implement cache management
   - Optimize memory usage

### Timeline

Q1 2024:
- Contrastive learning implementation
- Expert pruning and routing improvements
- Flash Attention V2 upgrade

Q2 2024:
- KV cache implementation
- Position understanding enhancements
- Memory optimization

Q3 2024:
- Sliding window attention
- Long sequence optimizations
- Performance tuning

## License

Vlabs License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{blue-blues,
  title = {Blues},
  author = {vaibhavg Gavade},
  year = {2024},
  url = {https://github.com/yourusername/blues-moe}
}
```
