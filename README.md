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
    vocab_size=100250,      # Vocabulary size
    block_size=512          # Maximum sequence length
)
```

<<<<<<< HEAD
<<<<<<< HEAD
## TODO List & Implementation Roadmap

### 1. Representation Learning
- [ ] Implement contrastive learning for embeddings
  - [ ] Basic pairwise contrastive loss
  - [ ] Multi-positive example support
  - [ ] Temperature-scaled InfoNCE loss
  - [ ] Hard negative mining
  - [ ] Data augmentation strategies

### 2. Expert System Improvements
- [ ] Enhanced Mixture of Experts
  - [ ] Dynamic expert pruning
  - [ ] Conditional computation paths
  - [ ] Expert specialization metrics
  - [ ] Adaptive routing strategies
  - [ ] Load balancing optimization
  - [ ] Expert dropout implementation

### 3. Position Understanding
- [ ] Advanced RoPE Enhancements
  - [ ] Dynamic RoPE scaling
  - [ ] Position interpolation
  - [ ] NTK-aware scaling
  - [ ] Position aliasing reduction
  - [ ] Extended context support
  - [ ] Position caching

### 4. Long Sequence Processing
- [ ] Implement sliding window attention
- [ ] Add sparse attention patterns
- [ ] Develop local-global attention mixing
- [ ] Create adaptive context window
- [ ] Optimize memory-efficient processing
- [ ] Implement progressive chunking

### 5. Attention Optimization
- [ ] Upgrade to Flash Attention V2
  - [ ] Automatic precision switching
  - [ ] Custom CUDA kernels
  - [ ] Memory access optimization
  - [ ] Attention pattern pruning
  - [ ] Pattern caching

### 6. Memory Efficient Inference
- [ ] Advanced KV cache system
  - [ ] Quantized cache storage
  - [ ] Dynamic cache pruning
  - [ ] Prefetch optimization
  - [ ] Cache compression
  - [ ] Memory-mapped support

### 7. Code LLM Training Optimization
- [ ] Code-Specific Performance Enhancements
  - [ ] AST-aware tokenization optimization
  - [ ] Code structure caching system
  - [ ] Syntax-aware attention masks
  - [ ] Language-specific expert routing
  - [ ] Multi-language token optimization

- [ ] Training Data Processing
  - [ ] Parallel code preprocessing pipeline
  - [ ] Efficient code repository streaming
  - [ ] Git-aware incremental updates
  - [ ] Smart code deduplication
  - [ ] Syntax tree batching

- [ ] Code Generation Optimization
  - [ ] Tree-structured beam search
  - [ ] Syntax-guided generation
  - [ ] Type-aware sampling
  - [ ] Context-aware completion cache
  - [ ] Code structure prefetching

- [ ] Performance Monitoring
  - [ ] Language-specific metrics tracking
  - [ ] Code quality evaluation pipeline
  - [ ] Syntax correctness verification
  - [ ] Real-time performance profiling
  - [ ] Memory usage optimization

- [ ] Hardware Utilization
  - [ ] Code-specific tensor optimizations
  - [ ] Custom CUDA kernels for AST processing
  - [ ] Efficient tree operation primitives
  - [ ] Parallel syntax parsing
  - [ ] GPU-accelerated code analysis

## Priority Matrix

| Feature | Impact | Difficulty | Priority |
|---------|--------|------------|----------|
| Contrastive Learning | High | Medium | 1 |
| Expert Pruning | High | Medium | 2 |
| Flash Attention V2 | High | Low | 3 |
| KV Cache | High | Medium | 4 |
| Sliding Window | Medium | High | 5 |
| Dynamic RoPE | Medium | Low | 6 |
| Code AST Optimization | High | High | 2 |
| Syntax-Aware Attention | High | Medium | 3 |
| Tree Operation CUDA | High | High | 2 |
| Code Cache System | Medium | Medium | 4 |

## Implementation Notes

### Contrastive Learning Strategy
- Begin with pairwise contrastive loss
- Gradually incorporate multi-positive support
- Implement temperature scaling
- Add sophisticated data augmentation

### Expert System Enhancement
- Implement utilization monitoring
- Develop adaptive routing mechanisms
- Optimize load balancing algorithms
- Add expert specialization metrics

### Position Understanding
- Benchmark various RoPE configurations
- Optimize for extended sequence lengths
- Implement position caching
- Reduce position aliasing

### Sequence Processing Optimization
- Start with basic sliding window
- Gradually add attention sparsity
- Implement efficient chunking
- Optimize memory usage patterns

### Attention System Updates
- Integrate Flash Attention V2
- Profile and optimize CUDA kernels
- Implement precision autotuning
- Add pattern caching mechanisms

### Inference Optimization
- Design efficient KV cache
- Implement compression strategies
- Optimize cache management
- Reduce memory footprint

## Training
=======
## Installation
>>>>>>> 47b3f217d181ac11672a37f8dd92db4c5da8b3d8
=======
## TODO List & Implementation Status

### Completed Features ✅
- [x] Basic model architecture implementation
- [x] Multi-Query Attention (MQA) support
- [x] Basic Mixture of Experts (MoE) system
- [x] Flash Attention integration
- [x] Gradient checkpointing
- [x] Basic contrastive learning
- [x] Dynamic batch sizing
- [x] Memory-efficient attention

### In Progress 🚧
- [ ] Contrastive Learning Improvements
  - [x] Basic pairwise contrastive loss
  - [x] Token replacement strategy
  - [ ] Multiple positive examples support
  - [ ] Better augmentation strategies
  - [ ] Improved similarity metrics

- [ ] Expert System Enhancements
  - [x] Basic expert routing
  - [ ] Dynamic expert pruning
  - [ ] Load balancing optimization
  - [ ] Expert specialization tracking
  - [ ] Adaptive routing thresholds

### Upcoming Features 📅
- [ ] Advanced Memory Management
  - [ ] KV cache implementation
  - [ ] Quantized cache storage
  - [ ] Smart memory swapping
  - [ ] Cache compression

- [ ] Position Understanding
  - [ ] Dynamic RoPE scaling
  - [ ] NTK-aware position embeddings
  - [ ] Extended context support
  - [ ] Position aliasing reduction

- [ ] Model Optimization
  - [ ] Flash Attention V2 upgrade
  - [ ] Model quantization
  - [ ] Better checkpoint management
  - [ ] Custom CUDA kernels

- [ ] Training Improvements
  - [ ] Progressive learning rates
  - [ ] Custom loss functions
  - [ ] Better error recovery
  - [ ] Training metrics dashboard

### Priority Matrix

| Feature | Impact | Difficulty | Priority |
|---------|---------|------------|-----------|
| Contrastive Improvement | High | Medium | 1 |
| Memory Management | High | High | 2 |
| Expert Enhancement | High | Medium | 3 |
| Position System | Medium | Medium | 4 |
| Flash Attn V2 | High | Low | 5 |

## Installation
>>>>>>> 693e12f (contrastive basic)

1. Clone the repository:
```bash
git clone https://github.com/blue-blues/blues-1
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

## High Performance Training

### Hardware Optimization

#### GPU Memory Management
- Dynamic batch size adjustment
- Gradient accumulation with automatic scaling
- Memory-efficient gradient checkpointing
- Smart memory swapping between CPU/GPU
- Automatic mixed precision (AMP) training
- Gradient clipping with adaptive thresholds

#### Multi-GPU Optimization
- Pipeline parallelism for large models
- Tensor parallelism across GPUs
- Efficient all-reduce operations
- Dynamic load balancing
- Zero redundancy optimizer (ZeRO) stages
- Sharded data parallelism

### Training Optimizations

#### Computational Efficiency
```python
training_config = {
    'compile_mode': 'reduce-overhead',  # or 'max-autotune'
    'dtype_policy': 'mixed_float16',
    'cudnn_benchmark': True,
    'gradient_accumulation_steps': 8,
    'cpu_offload': {
        'optimizer_state': True,
        'param_persistence': 'auto'
    }
}
```

#### Memory Management
```python
memory_config = {
    'pin_memory': True,
    'max_cached_batches': 16,
    'prefetch_factor': 2,
    'num_workers': 4,
    'persistent_workers': True
}
```

### Performance Tuning Guidelines

#### Batch Size Optimization
- Start with power of 2 batch sizes
- Use gradient accumulation for effective batch scaling
- Monitor memory usage vs. throughput
- Adjust based on validation metrics

#### Memory-Performance Trade-offs
1. **High Performance Mode**
   ```python
   config = {
       'precision': 'amp_bf16',
       'zero_stage': 3,
       'offload_optimizer': False,
       'pipeline_parallel': True
   }
   ```

2. **Memory Efficient Mode**
   ```python
   config = {
       'precision': 'fp16',
       'zero_stage': 2,
       'offload_optimizer': True,
       'activation_checkpointing': True
   }
   ```

### Advanced Training Features

#### Automatic Optimization
- Dynamic loss scaling
- Adaptive learning rate scheduling
- Automatic batch size finder
- Hardware-aware parameter tuning

#### Monitoring and Profiling
```bash
# Profile training performance
python train.py --profile --trace-export="timeline.json"

# Memory usage analysis
python train.py --memory-profile --report-interval=100
```

#### Optimization Examples

1. **Maximum Speed Setup**:
```bash
deepspeed train.py \
    --zero-stage 3 \
    --bf16 \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 2 \
    --partition-activations \
    --fast-mode
```

2. **Memory Efficient Setup**:
```bash
deepspeed train.py \
    --zero-stage 2 \
    --fp16 \
    --offload-optimizer \
    --gradient-clipping 1.0 \
    --partition-activations
```

### Custom Training Optimizations

1. **Custom Memory Manager**
```python
class MemoryManager:
    def __init__(self):
        self.reserved_memory = 0.2  # 20% memory reserve
        self.batch_multiplier = 1.0
        
    def adjust_batch_size(self, current_usage):
        # Dynamic batch size adjustment logic
        if current_usage > 0.85:  # 85% memory threshold
            self.batch_multiplier *= 0.8
        elif current_usage < 0.65:  # 65% memory threshold
            self.batch_multiplier *= 1.2
```

2. **Performance Monitoring**
```python
class PerformanceTracker:
    def __init__(self):
        self.throughput_history = []
        self.memory_usage = []
        
    def log_metrics(self, batch_time, mem_allocated):
        self.throughput_history.append(1.0 / batch_time)
        self.memory_usage.append(mem_allocated)
```

## Contributing

We welcome contributions! Please check our TODO list for areas that need attention.

1. Fork the repository
2. Create your feature branch
3. Implement your changes
4. Submit a pull request

Please ensure your code follows our style guidelines and includes appropriate tests.

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
