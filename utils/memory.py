import torch
import gc
from contextlib import contextmanager

class MemoryManager:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def clear_memory(self):
        """Clear unused memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @contextmanager
    def track_memory(self):
        """Context manager to track memory usage"""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            yield
        finally:
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                print(f"Peak memory usage: {peak_memory:.2f}MB")
                self.clear_memory()

class MemoryEfficientLoader:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
    
    def load_data(self, filepath):
        """Load data in memory-efficient manner"""
        try:
            return torch.load(filepath, map_location='cpu')
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

@contextmanager
def efficient_tensor_handling(tensors):
    """Context manager for efficient tensor operations"""
    try:
        yield
    finally:
        for tensor in tensors.values():
            if hasattr(tensor, 'detach'):
                del tensor

def estimate_memory_usage(model, input_shape):
    """Estimate memory usage for model with given input shape"""
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = total_params * 4  # Assuming float32
    
    # Estimate activation memory
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    est_activation_memory = batch_size * seq_len * model.config.n_embd * 4
    
    total_memory = (param_memory + est_activation_memory) / 1024**2  # Convert to MB
    return total_memory

def compute_optimal_chunk_size(available_memory, embedding_dim):
    """Compute optimal chunk size based on available memory"""
    # Assume 4 bytes per float
    bytes_per_token = embedding_dim * 4
    # Use 70% of available memory to be safe
    safe_memory = available_memory * 0.7
    return int(safe_memory / bytes_per_token)
