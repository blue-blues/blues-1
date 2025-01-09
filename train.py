import os
import torch
<<<<<<< HEAD
=======
import torch.nn.functional as F  # Add this import
>>>>>>> 693e12f (contrastive basic)
import time
import json
from model import blues
from config import config
from data import get_batch, encoder
from setting import *
from tqdm import tqdm
from optimize import optimize_model_for_training, MemoryOptimizer
from utils.memory import MemoryManager, efficient_tensor_handling, estimate_memory_usage
from utils.model_utils import estimate_loss  # Add this import

# Define the tokenizer
tokenizer = encoder

def verify_gpu_requirements():
    """Verify GPU requirements and available optimizations"""
    requirements = {
        'cuda': False,
        'flash_attn': False,
        'deepspeed': False,
        'cuda_version': None,
        'gpu_name': None,
        'gpu_memory': 0
    }
    
    try:
        if torch.cuda.is_available():
            requirements['cuda'] = True
            requirements['cuda_version'] = torch.version.cuda
            requirements['gpu_name'] = torch.cuda.get_device_name(0)
            requirements['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Check Flash Attention
            try:
                import flash_attn
                requirements['flash_attn'] = True
            except ImportError:
                print("Flash Attention not available. Using standard attention.")
            
            # Check DeepSpeed
            try:
                import deepspeed
                requirements['deepspeed'] = True
            except ImportError:
                print("DeepSpeed not available. Using standard training.")
            
            print("\nGPU Configuration:")
            print(f"CUDA Version: {requirements['cuda_version']}")
            print(f"GPU: {requirements['gpu_name']}")
            print(f"GPU Memory: {requirements['gpu_memory']:.2f}GB")
            print(f"Flash Attention: {'Available' if requirements['flash_attn'] else 'Not Available'}")
            print(f"DeepSpeed: {'Available' if requirements['deepspeed'] else 'Not Available'}")
        else:
            print("\nNo GPU detected. Running on CPU.")
    
    except Exception as e:
        print(f"Error checking GPU requirements: {e}")
    
    return requirements

def check_flash_attention_compatibility():
    """Check if Flash Attention is compatible with current GPU"""
    if not torch.cuda.is_available():
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    
    # Flash Attention requires Ampere (compute capability >= 8.0) or newer
    is_compatible = compute_capability[0] >= 8
    
    if not is_compatible:
        print(f"\nWarning: Flash Attention is not compatible with {gpu_name}")
        print(f"Compute capability: {compute_capability[0]}.{compute_capability[1]}")
        print("Flash Attention requires Ampere (RTX 30xx, A100) or newer GPUs.")
        print("Falling back to standard attention...")
    
    return is_compatible

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint file integrity"""
    try:
        # Try loading just the metadata first
        if not os.path.exists(checkpoint_path):
            return False, "Checkpoint file does not exist"
            
        if os.path.getsize(checkpoint_path) == 0:
            return False, "Checkpoint file is empty"
            
        # Try loading with weights_only first
        try:
            torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            return True, None
        except Exception:
            # If weights_only fails, try legacy loading with strict checks
            try:
                with open(checkpoint_path, 'rb') as f:
                    magic_number = f.read(2)
                    if magic_number != b'PK':  # ZIP file magic number
                        return False, "Invalid checkpoint file format"
                return True, None
            except Exception as e:
                return False, f"Checkpoint verification failed: {str(e)}"
                
    except Exception as e:
        return False, f"Error verifying checkpoint: {str(e)}"

def setup_training(use_gpu=True, use_deepspeed=True, resume_from=None):
    """Setup training with configurable GPU and DeepSpeed support and checkpoint resuming"""
    # Verify GPU requirements first
    gpu_check = verify_gpu_requirements()
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Initialize checkpoint variables
    start_iter = 0
    best_loss = float('inf')
    checkpoint = None
    
    # Initialize model and ensure consistent dtype
    model = blues(config, tokenizer).to(device)
    model = model.to(torch.float32)
    
    # Load checkpoint if resuming
    if resume_from:
        is_valid, error_msg = verify_checkpoint(resume_from)
        if is_valid:
            try:
                print(f"Loading checkpoint from: {resume_from}")
                checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_iter = checkpoint.get('iter', -1) + 1
                best_loss = checkpoint.get('loss', float('inf'))
                print(f"Successfully loaded checkpoint from iteration {start_iter-1}")
                print(f"Best loss: {best_loss:.4f}")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Starting fresh training...")
                checkpoint = None
        else:
            print(f"Invalid checkpoint file: {error_msg}")
            print("Starting fresh training...")
            
            # Try to find a valid backup checkpoint
            backup_files = ['latest.pt', 'best_model.pt']
            for backup in backup_files:
                backup_path = os.path.join(checkpoint_dir, backup)
                if os.path.exists(backup_path):
                    is_valid, _ = verify_checkpoint(backup_path)
                    if is_valid:
                        print(f"Found valid backup checkpoint: {backup}")
                        resume_from = backup_path
                        return setup_training(use_gpu, use_deepspeed, resume_from)
    
    # Check Flash Attention compatibility
    can_use_flash_attn = check_flash_attention_compatibility()
    
    # Update use flags based on availability
    use_gpu = use_gpu and gpu_check['cuda']
    use_deepspeed = use_deepspeed and gpu_check['deepspeed']
    use_flash_attn = can_use_flash_attn and gpu_check['flash_attn']
    
    # Update config with GPU feature availability
    config.update_gpu_settings(
        flash_attn_available=use_flash_attn,
        deepspeed_available=use_deepspeed
    )
    
    # Ensure all parameters are float32
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    
    # Apply optimizations
    model = optimize_model_for_training(model, config)
    
    if use_deepspeed:
        try:
            # Check available GPU memory before DeepSpeed init
            free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            if free_memory < 2 * 1024 * 1024 * 1024:  # Less than 2GB free
                print("Warning: Low GPU memory, falling back to standard training")
                raise RuntimeError("Insufficient GPU memory for DeepSpeed")

            import deepspeed
            # Set NCCL timeout to avoid hanging
            os.environ['NCCL_TIMEOUT'] = '30'  # 30 minutes
            os.environ['NCCL_BLOCKING_WAIT'] = '1'
            
            with open('ds_config.json') as f:
                ds_config = json.load(f)
            
            # Modify DeepSpeed config based on available memory
            ds_config['zero_optimization']['stage'] = 2  # Use more conservative setting
            ds_config['train_micro_batch_size_per_gpu'] = min(
                ds_config.get('train_micro_batch_size_per_gpu', 4),
                max(1, free_memory // (2 * 1024 * 1024 * 1024))  # Adjust based on free memory
            )
            
            model_engine, optimizer, _, _ = deepspeed.initialize(
                args=None,
                model=model,
                model_parameters=model.parameters(),
                config=ds_config,
                dist_init_required=True
            )
            print("Successfully initialized DeepSpeed")
            return model_engine, optimizer, start_iter, best_loss
            
        except Exception as e:
            print(f"DeepSpeed initialization failed: {str(e)}")
            print("Falling back to standard training...")
            use_deepspeed = False
            torch.cuda.empty_cache()
    
    # Standard optimizer fallback with proper settings initialization
    print("Using standard AdamW optimizer")
    
    # Define default optimizer settings if not present
    default_optimizer_settings = {
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01,
    }
    
    # Get optimizer settings, using defaults if not defined
    optimizer_settings = getattr(config, 'optimizer_config', {}).copy()
    optimizer_settings = {**default_optimizer_settings, **optimizer_settings}
    
    # Remove any DeepSpeed specific settings
    optimizer_settings.pop('capturable', None)
    optimizer_settings.pop('zero_optimization', None)
    
    # Adjust learning rate for standard training if needed
    if 'lr' in optimizer_settings:
        optimizer_settings['lr'] *= 0.5  # Reduce learning rate for stability
    else:
        optimizer_settings['lr'] = default_optimizer_settings['lr'] * 0.5
    
    print(f"Using optimizer settings: {optimizer_settings}")
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_settings)
    
    # Load optimizer state if checkpoint is available
    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Resumed optimizer state")
        except Exception as e:
            print(f"Error loading optimizer state: {e}")
            print("Starting with fresh optimizer...")
    
    return model, optimizer, start_iter, best_loss

<<<<<<< HEAD
def train_step(model, optimizer, x, y, use_deepspeed=True, scaler=None):
    """Memory efficient training step"""
    try:
        batch_size = x.size(0)
        if batch_size == 0:
            print("Warning: Received empty batch, skipping step")
            return 0.0
            
        # Ensure sub_batch_size is at least 1
        sub_batch_size = max(1, min(32, batch_size))
        total_loss = 0
        
        num_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size  # Ceiling division
        if num_sub_batches == 0:
            print("Warning: Invalid batch configuration")
            return 0.0
            
        for i in range(0, batch_size, sub_batch_size):
            end_idx = min(i + sub_batch_size, batch_size)
            sub_x = x[i:end_idx]
            sub_y = y[i:end_idx]
            
            if sub_x.size(0) == 0 or sub_y.size(0) == 0:
                print(f"Warning: Empty sub-batch detected at index {i}")
                continue
                
            # Update autocast to use new syntax
            try:
                if scaler is not None:
                    with torch.amp.autocast('cuda', enabled=True):
                        logits, loss = model(sub_x, sub_y)
                        loss = loss / num_sub_batches  # Use calculated num_sub_batches
                    scaler.scale(loss).backward()
                else:
                    logits, loss = model(sub_x, sub_y)
                    loss = loss / num_sub_batches  # Use calculated num_sub_batches
                    loss.backward()
                
                total_loss += loss.item()
                del logits, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except RuntimeError as e:
                print(f"Error processing sub-batch: {str(e)}")
                continue
        
        return total_loss
        
    except Exception as e:
        print(f"Error in train_step: {str(e)}")
        print(f"Batch size: {batch_size if 'batch_size' in locals() else 'unknown'}")
        print(f"Sub-batch size: {sub_batch_size if 'sub_batch_size' in locals() else 'unknown'}")
        raise e

=======
def train_step(model, optimizer, x, y, contrast_x=None, use_deepspeed=True, scaler=None):
    """Modified training step to handle contrastive learning"""
    try:
        batch_size = x.size(0)
        if batch_size == 0:
            return 0.0
        
        if scaler is not None:
            with torch.amp.autocast('cuda', enabled=True):
                logits, loss = model(x, y, contrast_x)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits, loss = model(x, y, contrast_x)
            loss = loss / gradient_accumulation_steps
            loss.backward()
        
        return loss.item() if loss is not None else 0.0
        
    except Exception as e:
        print(f"Error in train_step: {str(e)}")
        # Add stack trace for debugging if needed
        import traceback
        traceback.print_exc()
        raise e

def get_contrastive_batch(batch_x, model, device):
    """Generate contrastive pairs using model's tokenizer and embeddings"""
    contrast_x = batch_x.clone()
    batch_size, seq_len = contrast_x.shape
    
    # Get vocabulary size from model
    vocab_size = model.vocab_size
    
    # Get padding token if available, otherwise use 0
    # Ensure pad_token is an integer
    pad_token = getattr(model.config, 'pad_token_id', 0)
    if not isinstance(pad_token, int):
        pad_token = 0  # Fallback to 0 if pad_token is not an integer
    
    # Get embeddings for similarity-based token replacement
    with torch.no_grad():
        # Get token embeddings from model's embedding layer
        token_embeddings = model.embedder.weight.detach()
        # Normalize embeddings for cosine similarity
        normalized_embeddings = F.normalize(token_embeddings, p=2, dim=1)
    
    # Define augmentation strategies with probabilities
    aug_probs = torch.tensor([0.4, 0.3, 0.3], device=device)  # [replace_similar, local_shuffle, delete]
    
    for i in range(batch_size):
        # Choose augmentation strategy
        strategy = torch.multinomial(aug_probs, 1).item()
        
        # Create mask tensor for non-padding tokens using integer comparison
        mask = (contrast_x[i] != pad_token)
        mask = mask.to(device)  # Move mask to device after creation
        valid_positions = torch.nonzero(mask).squeeze(-1)
        
        if valid_positions.numel() == 0:
            continue
            
        # Number of positions to augment
        num_aug = max(1, int(valid_positions.numel() * 0.15))  # Augment 15% of tokens
        perm = torch.randperm(valid_positions.numel(), device=device)
        aug_positions = valid_positions[perm[:num_aug]]
        
        if strategy == 0:  # replace_similar
            for pos in aug_positions:
                orig_token = contrast_x[i, pos]
                orig_embedding = normalized_embeddings[orig_token].to(device)
                similarities = torch.matmul(normalized_embeddings.to(device), orig_embedding)
                # Exclude the original token from similar tokens
                similarities[orig_token] = -float('inf')
                similar_tokens = torch.topk(similarities, k=5, dim=0)[1]
                new_token = similar_tokens[torch.randint(0, 5, (1,), device=device)]
                contrast_x[i, pos] = new_token
                
        elif strategy == 1:  # local_shuffle
            for pos in aug_positions:
                window_start = max(0, pos - 2)
                window_end = min(seq_len, pos + 3)
                window = contrast_x[i, window_start:window_end]
                window_mask = (window != pad_token)
                window_mask = window_mask.to(device)
                valid_window_indices = torch.nonzero(window_mask).squeeze(-1)
                if valid_window_indices.numel() > 1:
                    perm = torch.randperm(valid_window_indices.numel(), device=device)
                    shuffled = window[valid_window_indices][perm]
                    window[valid_window_indices] = shuffled
                    contrast_x[i, window_start:window_end] = window
                        
        else:  # delete
            # Sort positions in descending order to avoid shifting issues
            for pos in sorted(aug_positions.tolist(), reverse=True):
                contrast_x[i, pos:-1] = contrast_x[i, pos+1:].clone()
                contrast_x[i, -1] = pad_token
    
    return contrast_x

>>>>>>> 693e12f (contrastive basic)
def verify_data_cache():
    """Verify that preprocessed data exists"""
    metadata_file = os.path.join(data_cache_dir, 'metadata.pkl')
    if not os.path.exists(data_cache_dir) or not os.path.exists(metadata_file):
        print("Preprocessed data not found!")
        print("Please run 'python data.py' first to preprocess the dataset.")
        return False
    return True

def verify_model_init(model, optimizer):
    """Verify model and optimizer initialization"""
    try:
        # Check model parameters and dtype
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                print(f"Warning: Parameter {name} has dtype {param.dtype}")
                param.data = param.data.to(torch.float32)
        
        first_param = next(model.parameters())
        param_dtype = first_param.dtype
        device = first_param.device
        print(f"Model dtype: {param_dtype}")
        print(f"Model device: {device}")
        print(f"Model training mode: {model.training}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {num_params:,}")
        
        # Verify optimizer
        param_groups = len(optimizer.param_groups)
        print(f"Optimizer param groups: {param_groups}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Test forward pass with small batch using correct dtypes
        test_x = torch.randint(0, config.vocab_size, (2, 4), device=device, dtype=torch.long)  # Changed to long
        test_y = torch.randint(0, config.vocab_size, (2, 4), device=device, dtype=torch.long)  # Changed to long
        
        with torch.no_grad():
            model.eval()  # Temporarily set to eval mode
            logits, loss = model(test_x, test_y)
            print(f"Test forward pass - Loss: {loss.item():.4f}")
            model.train()  # Set back to training mode
            
            # Check output dtypes
            print(f"Logits dtype: {logits.dtype}")
            print(f"Loss dtype: {loss.dtype}")
            
            # Verify embedding input/output shapes
            print(f"Input shape: {test_x.shape}")
            print(f"Logits shape: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"Model verification failed: {str(e)}")
        print(f"Debug info:")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Test input dtype: {test_x.dtype if 'test_x' in locals() else 'not created'}")
        return False

class TrainingLogger:
    def __init__(self, eval_interval=500):
        self.eval_interval = eval_interval
        self.start_time = time.time()
        self.best_loss = float('inf')
        
    def log_eval(self, iter_num, train_loss, val_loss, lr, total_iters):
        elapsed = time.time() - self.start_time
        hours_per_iter = elapsed / (iter_num + 1) / 3600
        eta = hours_per_iter * (total_iters - iter_num)
        
        print(f"\n{'='*50}")
        print(f"Iteration {iter_num}/{total_iters}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Learning rate: {lr:.2e}")
        print(f"Time elapsed: {elapsed/3600:.2f}h")
        print(f"ETA: {eta:.2f}h")
        print(f"{'='*50}")
        
        # Track best loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            return True
        return False

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
    gpu_check = verify_gpu_requirements()
    use_deepspeed = use_gpu and gpu_check['deepspeed'] and flash_optimizations['use_flash_attn']
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    print(f"\nTraining Configuration:")
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")
    if use_gpu:
        print(f"Attention: {'Flash' if gpu_check['flash_attn'] else 'Standard'}")
        print(f"Training: {'DeepSpeed' if use_deepspeed else 'Standard'}")
    
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
    
    # Setup training environment with explicit dtype and device handling
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Check for interrupted checkpoint with validation
    resume_from = None
    checkpoint_files = ['interrupted.pt', 'latest.pt', 'best_model.pt']
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        if os.path.exists(checkpoint_path):
            is_valid, _ = verify_checkpoint(checkpoint_path)
            if is_valid:
                resume_from = checkpoint_path
                print(f"Found valid checkpoint: {checkpoint_file}")
                break
            else:
                print(f"Found corrupt checkpoint: {checkpoint_file}, skipping...")
    
    try:
        model, optimizer, start_iter, best_loss = setup_training(
            use_gpu=use_gpu, 
            use_deepspeed=use_deepspeed,
            resume_from=resume_from
        )
    except Exception as e:
        print(f"Error during training setup: {str(e)}")
        print("Attempting to recover with minimal configuration...")
        # Try one more time with minimal settings
        use_deepspeed = False
        torch.cuda.empty_cache()
        model, optimizer, start_iter, best_loss = setup_training(
            use_gpu=True, 
            use_deepspeed=False,
            resume_from=resume_from
        )
    
    model = model.to(device)  # Ensure model is on correct device
    
    # Initialize scaler for mixed precision training
    scaler = None
    if use_gpu and not use_deepspeed:
        scaler = torch.cuda.amp.GradScaler()
        print("Initialized mixed precision training with GradScaler")
    
    # Ensure model is in the correct dtype and device
    model = model.float().to(device)
    
    # Verify model initialization
    if not verify_model_init(model, optimizer):
        print("Model initialization failed!")
        exit(1)
    
    # Set model to training mode explicitly
    model.train()
    
    # Enable garbage collection
    import gc
    gc.enable()
    
    # Memory manager (moved after device initialization)
    memory_manager = MemoryManager(device)
    
    # Estimate memory requirements
    sample_batch = get_batch('train', 1, max_seq_len, device)
    estimated_memory = estimate_memory_usage(model, sample_batch[0].shape)
    print(f"Estimated memory usage per batch: {estimated_memory:.2f}MB")
    
    # Initialize logger
    logger = TrainingLogger(eval_interval=eval_interval)
    logger.best_loss = best_loss
    
    print("\nStarting training...")
    try:
        running_loss = 0.0
        accumulated_grads = 0
        
        # Update progress bar to start from resumed iteration
        progress = tqdm(
            range(start_iter, max_iters), 
            desc="Training", 
            ncols=80, 
            leave=True,
            initial=start_iter,
            total=max_iters
        )
        
        for iter_num in progress:
            try:
                # Ensure valid batch size
                effective_batch_size = max(1, current_batch_size // gradient_accumulation_steps)
                
                # Gradient accumulation loop
                optimizer.zero_grad()
                total_loss = 0
                valid_steps = 0
                
                for accum_step in range(gradient_accumulation_steps):
                    try:
                        x, y = get_batch('train', effective_batch_size, max_seq_len, device)
<<<<<<< HEAD
                        
                        # Validate batch data
                        if x.size(0) == 0 or y.size(0) == 0:
                            print(f"Warning: Skipping empty batch in accumulation step {accum_step}")
                            continue
                            
=======
                        # Generate contrastive pairs
                        contrast_x = get_contrastive_batch(x, model, device)
                        
>>>>>>> 693e12f (contrastive basic)
                        loss = train_step(
                            model=model,
                            optimizer=optimizer,
                            x=x,
                            y=y,
<<<<<<< HEAD
=======
                            contrast_x=contrast_x,
>>>>>>> 693e12f (contrastive basic)
                            use_deepspeed=use_deepspeed,
                            scaler=scaler
                        )
                        
                        if loss > 0:  # Only count valid losses
                            total_loss += loss
                            valid_steps += 1
                            accumulated_grads += 1
                            
                        del x, y
                        
                    except RuntimeError as e:
                        print(f"Error in accumulation step {accum_step}: {str(e)}")
                        continue
                
                # Only update if we had valid steps
                if valid_steps > 0:
                    if accumulated_grads >= gradient_accumulation_steps:
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        accumulated_grads = 0
                    
                    # Update running loss
                    avg_loss = total_loss / valid_steps
                    running_loss = 0.9 * running_loss + 0.1 * avg_loss
                    
                    # Update progress bar
                    progress.set_postfix({
                        'loss': f'{running_loss:.4f}',
                        'batch_size': effective_batch_size
                    })
                
                # Evaluation
                if iter_num > 0 and iter_num % eval_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        val_loss = evaluate_model(model, device, current_batch_size)
                    model.train()
                    
                    # Log evaluation results
                    is_best = logger.log_eval(
                        iter_num=iter_num,
                        train_loss=running_loss,
                        val_loss=val_loss,
                        lr=optimizer.param_groups[0]['lr'],
                        total_iters=max_iters
                    )
                    
                    # Save if best
                    if is_best:
                        save_checkpoint(model, optimizer, iter_num, val_loss, 'best_model.pt')
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    current_batch_size = max(min_batch_size, current_batch_size // 2)
                    print(f"\nOOM error - reducing batch size to {current_batch_size}")
                    continue
                raise e
                
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving checkpoint...")
        save_checkpoint(model, optimizer, iter_num, running_loss, 'interrupted.pt')
    
    finally:
        progress.close()
        # Save both final and latest checkpoints
        save_checkpoint(model, optimizer, iter_num, running_loss, 'final_model.pt')
        save_checkpoint(model, optimizer, iter_num, running_loss, 'latest.pt')

def evaluate_model(model, device, batch_size, num_batches=5):
    """Evaluate model on validation set"""
    total_loss = 0
    for _ in range(num_batches):
        try:
            x, y = get_batch('val', batch_size, model.max_seq_len, device)
            _, loss = model(x, y)
            total_loss += loss.item()
        except Exception as e:
            print(f"Error during evaluation: {e}")
            continue
    return total_loss / num_batches

def save_checkpoint(model, optimizer, iter, loss, filename):
    """Save model checkpoint with training state"""
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': iter,
            'loss': loss,
            'random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'timestamp': time.time()
        }, os.path.join(checkpoint_dir, filename))
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

if __name__ == "__main__":
    main()
