# Blues: Mixture of Experts Language Model

Blues is a language model that combines Multi-Query Attention (MQA) with Mixture of Experts (MoE) architecture, designed for efficient and scalable natural language processing.

## Architecture Overview

### Key Components

1. **Multi-Query Attention (MQA)**
   - Supports different numbers of attention heads for queries and key-values
   - Uses rotary positional embeddings (RoPE)
   - Configurable number of attention heads and key-value heads
   - Optional Flash Attention support for improved performance

2. **Mixture of Experts (MoE)**
   - Dynamic routing between experts
   - Configurable number of total experts and active experts per token
   - Gated expert networks with GeLU activation
   - Noise-based exploration during training

3. **Model Structure**
   - Embedding layer with scaled initialization
   - Multiple decoder layers with MQA and MoE
   - RMSNorm for layer normalization
   - Shared embedding weights for input and output

### Technical Specifications

```python
# Default Configuration
vocab_size = 200019
max_position_embeddings = 256
num_layers = 4
hidden_size = 192
head_dim = 48
num_attention_heads = 4
num_key_value_heads = 2
tot_num_experts = 4
chosen_num_experts = 1
```

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

2. Prepare your dataset in CSV format with a 'Text' column.

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
