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

## License

This project is open-source and available under the MIT License.
