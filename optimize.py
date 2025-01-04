import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from setting import flash_optimizations
from contextlib import contextmanager

class MemoryOptimizer:
    @staticmethod
    def compute_optimal_batch_size(model, sample_input, min_batch=1, max_batch=128):
        """Binary search for optimal batch size"""
        left, right = min_batch, max_batch
        optimal = min_batch
        
        while left <= right:
            mid = (left + right) // 2
            try:
                # Scale input to test batch size
                test_input = sample_input.repeat(mid // sample_input.shape[0] + 1, 1)[:mid]
                
                # Ensure model is in float32
                model = model.float()
                
                # Convert test input to proper dtypes
                if isinstance(test_input, tuple):
                    test_input = tuple(t.long() if i == 0 else t.float() 
                                     for i, t in enumerate(test_input))
                else:
                    test_input = test_input.long()
                
                with torch.no_grad():
                    model(test_input)
                left = mid + 1
                optimal = mid
            except RuntimeError as e:
                if "out of memory" in str(e):
                    right = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        # Return slightly smaller batch size for safety
        return max(1, int(optimal * 0.95))

def optimize_model_for_training(model, config):
    """Apply various optimizations to the model"""
    if flash_optimizations['use_flash_attn'] and torch.cuda.is_available():
        try:
            from flash_attn import flash_attn_func
            model = enable_flash_attention(model)
            print("Flash attention enabled successfully")
        except ImportError:
            print("Flash attention package not available, using standard attention")
    elif not torch.cuda.is_available():
        print("Running on CPU, using standard attention")
    
    if flash_optimizations['mem_efficient']:
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        # Optimize memory usage
        for module in model.modules():
            if hasattr(module, 'set_memory_efficient_attention'):
                module.set_memory_efficient_attention(True)
    
    if flash_optimizations['use_cuda_fp16']:
        model = model.half()
    
    return model

def enable_flash_attention(model):
    """Enable flash attention for supported layers"""
    flash_enabled = False
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module._flash_attention = True
            flash_enabled = True
    
    if not flash_enabled:
        print("No compatible attention layers found for flash attention")
    return model

class ModelOptimizations:
    @staticmethod
    def apply_mixed_precision(model, optimizer):
        """Setup mixed precision training"""
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        return scaler
    
    @staticmethod
    def optimize_memory_usage(model):
        """Apply memory optimizations"""
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Use memory efficient attention if available
        for module in model.modules():
            if hasattr(module, 'set_memory_efficient_attention'):
                module.set_memory_efficient_attention(True)
        
        return model
    
    @staticmethod
    def prepare_model_for_kbit_training(model):
        """Prepare model for 8-bit or 4-bit training"""
        try:
            import bitsandbytes as bnb
            from bitsandbytes.optim import AdamW8bit
            
            # Convert applicable linear layers to 8-bit
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module = bnb.nn.Linear8bitLt.from_float(module)
            
            return model
        except ImportError:
            print("bitsandbytes not available, skipping 8-bit optimization")
            return model

@contextmanager
def optimize_model_for_inference(model):
    """Context manager to optimize model for inference"""
    # Store original training state
    training = model.training
    
    try:
        # Set to eval mode
        model.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            # Optimize memory usage
            if hasattr(model, 'config'):
                # Disable dropout
                model.config.dropout = 0
                
                # Enable memory efficient attention if available
                if hasattr(model.config, 'mem_efficient'):
                    model.config.mem_efficient = True
            
            # Fuse attention operations if possible
            for module in model.modules():
                if hasattr(module, 'enable_fused_attention'):
                    module.enable_fused_attention()
            
            yield model
            
    finally:
        # Restore original training state
        model.train(training)
        if hasattr(model, 'config'):
            # Restore original dropout
            model.config.dropout = getattr(model.config, '_original_dropout', 0.1)
