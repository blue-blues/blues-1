import os
import torch
import time
import json
from model import blues
from config import config
from data import get_batch, encoder
from setting import *
from tqdm import tqdm  # Add progress bar
from optimize import optimize_model_for_training, MemoryOptimizer
from utils.memory import MemoryManager, efficient_tensor_handling, estimate_memory_usage

# Define the tokenizer
tokenizer = encoder

def setup_training(use_gpu=True, use_deepspeed=True):
    """Setup training with configurable GPU and DeepSpeed support"""
    # Set special tokens in config
    from data import PAD_TOKEN, START_TOKEN, END_TOKEN
    config.set_special_tokens(PAD_TOKEN, START_TOKEN, END_TOKEN)
    
    model = blues(config, tokenizer)
    
    # Apply optimizations
    model = optimize_model_for_training(model, config)
    
    # Compute optimal batch size
    if use_gpu:
        sample_input, _ = get_batch('train', 1, config.max_position_embeddings, 'cuda')
        optimal_batch_size = MemoryOptimizer.compute_optimal_batch_size(model, sample_input)
        print(f"Computed optimal batch size: {optimal_batch_size}")
        global batch_size
        batch_size = min(batch_size, optimal_batch_size)
    
    if use_gpu and torch.cuda.is_available():
        if use_deepspeed:
            import deepspeed
            with open('ds_config.json') as f:
                ds_config = json.load(f)
            model_engine, optimizer, _, _ = deepspeed.initialize(
                args=None,
                model=model,
                model_parameters=model.parameters(),
                config=ds_config,
                dist_init_required=True
            )
            return model_engine, optimizer
        else:
            model = model.cuda()
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
            return model, optimizer
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
        return model, optimizer

def train_step(model, optimizer, x, y, use_deepspeed=True, scaler=None):
    """Single training step with DeepSpeed and mixed precision support"""
    if use_deepspeed:
        outputs = model(x, y)
        loss = outputs[1]
        model.backward(loss)
        model.step()
    else:
        if scaler is not None:  # Using mixed precision without DeepSpeed
            with torch.cuda.amp.autocast():
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
        else:  # CPU training
            logits, loss = model(x, y)
            loss.backward()
        return loss.item()

def verify_data_cache():
    """Verify that preprocessed data exists"""
    metadata_file = os.path.join(data_cache_dir, 'metadata.pkl')
    if not os.path.exists(data_cache_dir) or not os.path.exists(metadata_file):
        print("Preprocessed data not found!")
        print("Please run 'python data.py' first to preprocess the dataset.")
        return False
    return True

def main():
    # Check for preprocessed data before starting training
    if not verify_data_cache():
        exit(1)
    
    print("Found preprocessed data cache, proceeding with training...")
    
    # Make batch_size modifiable within the function
    global batch_size  # Allow modification of global batch_size if needed
    current_batch_size = batch_size  # Local copy for potential dynamic adjustment
    
    # Detect hardware and set training mode
    use_gpu = torch.cuda.is_available()
    use_deepspeed = use_gpu and flash_optimizations['use_flash_attn']
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    print(f"Training on: {'GPU' if use_gpu else 'CPU'}")
    print(f"Initial batch size: {current_batch_size}")
    print(f"Number of iterations: {max_iters}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Ensure data cache directory exists
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
        print(f"Created data cache directory: {data_cache_dir}")
    
    # Verify data processing before training
    print("Verifying data processing...")
    max_seq_len = min(config.block_size, config.max_position_embeddings)
    _ = get_batch('train', 1, max_seq_len, 'cpu')
    print("Data verification successful!")
    
    # Setup training environment
    model, optimizer = setup_training(use_gpu, use_deepspeed)
    scaler = torch.cuda.amp.GradScaler() if use_gpu and not use_deepspeed else None
    
    # Enable garbage collection
    import gc
    gc.enable()
    
    # Memory manager (moved after device initialization)
    memory_manager = MemoryManager(device)
    
    # Estimate memory requirements
    sample_batch = get_batch('train', 1, max_seq_len, device)
    estimated_memory = estimate_memory_usage(model, sample_batch[0].shape)
    print(f"Estimated memory usage per batch: {estimated_memory:.2f}MB")
    
    # Training loop
    start_time = time.time()
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if use_gpu else 'cpu')

    try:
        pbar = tqdm(range(max_iters), desc="Training")
        accumulated_loss = 0
        for iter in pbar:
            try:
                with memory_manager.track_memory():
                    # Zero gradients
                    if not use_deepspeed:
                        optimizer.zero_grad()
                    
                    # Gradient accumulation loop
                    for _ in range(gradient_accumulation_steps):
                        try:
                            x, y = get_batch('train', current_batch_size, config.max_position_embeddings, device)
                            
                            with efficient_tensor_handling({'x': x, 'y': y}):
                                # Training step
                                logits, loss = model(x, y)
                                # Scale loss for gradient accumulation
                                loss = loss / gradient_accumulation_steps
                                loss.backward()
                                
                                accumulated_loss += loss.item()
                                
                                # Clear memory
                                del x, y, logits
                                gc.collect()
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                current_batch_size = max(1, current_batch_size // 2)
                                print(f"\nWARNING: out of memory, reducing batch size to {current_batch_size}")
                                continue
                            else:
                                raise e
                    
                    # Optimizer step after accumulation
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Clear memory periodically
                    if iter % 100 == 0:
                        memory_manager.clear_memory()
                    
                    # Add chunk information to progress bar
                    if hasattr(datasets['train'], 'current_chunk_num'):
                        pbar.set_postfix({
                            'loss': f'{accumulated_loss:.4f}',
                            'batch_size': current_batch_size,
                            'chunk': datasets['train'].current_chunk_num
                        })
                    accumulated_loss = 0
                    
                    # Logging and checkpointing
                    if iter % eval_interval == 0:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        if use_deepspeed and not model.is_first_worker():
                            continue
                            
                        losses = estimate_loss(model, current_batch_size)
                        print(f"\nStep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {elapsed:.2f}s")
                        
                        # Save checkpoint
                        save_path = os.path.join(checkpoint_dir, f"model_iter_{iter}.pt")
                        if use_deepspeed:
                            model.save_checkpoint(checkpoint_dir, f"model_iter_{iter}")
                        else:
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'iter': iter,
                                'loss': losses,
                                'batch_size': current_batch_size
                            }, save_path)

            except Exception as e:
                print(f"\nError during training iteration {iter}: {e}")
                raise

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': iter,
            'loss': loss if 'loss' in locals() else None,
            'batch_size': current_batch_size
        }, os.path.join(checkpoint_dir, "interrupt_checkpoint.pt"))
        print("Checkpoint saved.")

if __name__ == "__main__":
    main()
