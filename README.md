# Blues-01: MoE Language Model

A PyTorch implementation of a Mixture of Experts (MoE) language model with multi-query attention and memory-efficient training.

## Features

- Mixture of Experts (MoE) architecture for efficient scaling
- Multi-Query Attention (MQA) for reduced memory usage
- Memory-efficient training with chunked datasets
- Tiktoken o200k_base tokenizer integration
- Dynamic batch sizing and gradient accumulation
- FlashAttention support (optional)
- RMSNorm and rotary position embeddings

## Architecture

- Vocab size: 200k (tiktoken o200k_base)
- Model dimensions: 768
- Attention heads: 12
- Layers: 12 (alternating MoE layers)
- Experts: 8 per MoE layer
- Expert routing: Top-2 routing with noise
- Position embeddings: Rotary (RoPE)

## Requirements

```
deepspeed>=0.10.0
flash-attn>=2.3.3
torch>=2.0.0
einops>=0.6.1
triton>=2.0.0
```

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

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```
```
python data.py --input_dir data/raw --cache_dir data_cache
```
3. Update configuration in `config.py` if needed.

### Running Training

For single GPU:
```bash
python train.py
```

For multi-GPU training with DeepSpeed:
```bash
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

## Working with Multiple Datasets

### Dataset Merging

The model supports training on multiple datasets from different sources. You can merge datasets using the provided tools:

```bash
# Merge multiple datasets with equal weights
python merge_data.py --datasets data_cache_user1 data_cache_user2 data_cache_user3

# Merge with custom weights (e.g., 70% first dataset, 30% second dataset)
python merge_data.py --datasets data_cache_user1 data_cache_user2 --weights 0.7 0.3
```

### Dataset Processing Workflow

1. **Preprocess Individual Datasets**:
```bash
# Process first dataset
python data.py --dataset dataset1.csv --data_cache_dir data_cache_1

# Process second dataset
python data.py --dataset dataset2.csv --data_cache_dir data_cache_2
```

2. **Merge Datasets**:
```bash
python merge_data.py --datasets data_cache_1 data_cache_2
```

3. **Train on Merged Data**:
```bash
python train.py --data_cache_dir data_cache/merged
```

### Merging Strategies

The system supports different merging strategies:

- `weighted`: Use custom weights for each dataset
- `equal`: Treat all datasets equally
- `proportional`: Weight by dataset sizes

Configure merging strategy in `config.json`:
```json
{
    "datasets": {
        "paths": ["data_cache_1", "data_cache_2"],
        "weights": [0.7, 0.3],
        "merge_strategy": "weighted"
    }
}
```

### Memory Efficient Processing

- Datasets are processed and stored in chunks
- Each chunk is loaded only when needed
- Automatic memory management
- Support for large-scale datasets

### Dataset Verification

You can verify merged datasets:
```bash
python merge_data.py --verify --datasets data_cache_1 data_cache_2
```
This will:
- Check data integrity
- Verify chunk consistency
- Calculate dataset statistics
- Report any issues

## Model Features

### 1. Multi-Query Attention
- Efficient attention mechanism with shared key-value heads
- RoPE positional embeddings for better position awareness
- Optional Flash Attention support for faster computation

### 2. Mixture of Experts
- Dynamic expert routing based on input content
- Noise-based exploration during training
- Load balancing through auxiliary loss

### 3. Architecture Optimizations
- Mixed precision training support
- DeepSpeed integration for distributed training
- Memory efficient attention implementation
- Gradient checkpointing option

## Generation

The model supports various text generation settings:

```python
generation_config = {
    'temperature': 0.8,
    'top_p': 0.9,
    'top_k': 50,
    'max_length': 100
}
```

## Generation and Evaluation

### Using the Generation Script

The `generate.py` script provides an easy interface for text generation and model evaluation:

```bash
# Basic text generation
python generate.py --checkpoint checkpoints/model_iter_1000.pt --prompt "Once upon a time"

# Advanced generation with custom parameters
python generate.py --checkpoint checkpoints/model_iter_1000.pt \
    --prompt "The future of AI is" \
    --max_length 200 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 40

# Evaluate perplexity on custom text
python generate.py --checkpoint checkpoints/model_iter_1000.pt \
    --eval_text "This is a test sentence to evaluate the model's perplexity."
```

### Generation Parameters

- `--checkpoint`: Path to the saved model checkpoint
- `--prompt`: Input text to start generation
- `--max_length`: Maximum number of tokens to generate
- `--temperature`: Controls randomness (higher = more random)
- `--top_p`: Nucleus sampling parameter
- `--top_k`: Top-k sampling parameter
- `--eval_text`: Text to calculate perplexity on

### Interactive Generation

You can also use the model programmatically:

```python
from generate import load_model, sample_text

model = load_model("checkpoints/model_iter_1000.pt")
text = sample_text(
    model,
    prompt="Write a story about",
    max_length=200,
    temperature=0.8,
    top_p=0.9,
    top_k=50
)
```

### Evaluation Metrics

The model supports several evaluation metrics:

1. **Perplexity**: Measures how well the model predicts a text sample
2. **Expert utilization**: Tracks the distribution of tokens across experts
3. **Training/validation loss**: Monitors model convergence
4. **Generation quality**: Subjective evaluation of generated text

## Monitoring and Checkpoints

- Regular evaluation during training
- Checkpoint saving at configured intervals
- Training loss and validation loss monitoring
- Expert utilization tracking

## Performance Optimization

- Flash Attention for faster attention computation
- DeepSpeed integration for distributed training
- Mixed precision training
- Memory optimization settings

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

This project is open-source and available under the MIT License.
