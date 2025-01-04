import os
import numpy as np
import tiktoken
import torch
from pathlib import Path
import csv
import pickle
from tqdm import tqdm
from setting import DEFAULT_SETTINGS

# Get settings with fallback values
dataset = DEFAULT_SETTINGS.get('dataset', "df_file.csv")
data_cache_dir = DEFAULT_SETTINGS.get('data_cache_dir', "data_cache")
chunk_size = DEFAULT_SETTINGS.get('chunk_size', 10000)

from utils.memory import MemoryManager, MemoryEfficientLoader, compute_optimal_chunk_size

# Initialize tiktoken encoder with o200k_base and special tokens
try:
    encoder = tiktoken.get_encoding("o200k_base")
    # Use tokens from within the vocab range
    START_TOKEN = encoder.n_vocab - 4  # Keep within vocab range
    END_TOKEN = encoder.n_vocab - 3
    PAD_TOKEN = encoder.n_vocab - 2
    
    # Verify tokens are within valid range
    assert all(0 <= token < encoder.n_vocab for token in [START_TOKEN, END_TOKEN, PAD_TOKEN]), \
        "Special tokens must be within vocabulary range"
    
    print(f"Initialized o200k_base tokenizer with vocabulary size: {encoder.n_vocab}")
    print(f"Special tokens - Start: {START_TOKEN}, End: {END_TOKEN}, Pad: {PAD_TOKEN}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize o200k_base tokenizer: {e}")

def save_chunk(data, filepath, chunk_num):
    """Save a chunk of tokenized data"""
    with open(filepath.format(chunk_num), 'wb') as f:
        pickle.dump(data, f)

def load_chunk(filepath, chunk_num):
    """Load a chunk of tokenized data"""
    with open(filepath.format(chunk_num), 'rb') as f:
        return pickle.load(f)

def process_and_save_data(filepath, cache_dir, chunk_size=1000):
    """Process the CSV in chunks and save tokenized data"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Define cache file patterns
    train_pattern = os.path.join(cache_dir, 'train_chunk_{}.pkl')
    val_pattern = os.path.join(cache_dir, 'val_chunk_{}.pkl')
    metadata_file = os.path.join(cache_dir, 'metadata.pkl')
    
    # Check if processing is already done
    if os.path.exists(metadata_file):
        with open(metadata_file, 'rb') as f:
            return pickle.load(f)
    
    print("Processing data in chunks and creating cache...")
    chunk_buffer = []
    total_tokens = 0
    chunk_num = 0
    
    # Count total rows first
    with open(filepath, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header row
    
    # Process file in chunks
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=total_rows, desc="Processing rows")
        
        for row in reader:
            if row.get('Text'):
                try:
                    # Add start and end tokens to the text
                    tokens = [START_TOKEN] + encoder.encode(row['Text']) + [END_TOKEN]
                    chunk_buffer.extend(tokens)
                    total_tokens += len(tokens)
                    
                    # Save chunk when buffer is full
                    if len(chunk_buffer) >= chunk_size:
                        is_train = (chunk_num < int(0.9 * total_rows))  # 90% for training
                        save_pattern = train_pattern if is_train else val_pattern
                        save_chunk(chunk_buffer, save_pattern, chunk_num)
                        chunk_buffer = []
                        chunk_num += 1
                        print(f"\rSaved chunk {chunk_num}", end="")
                        
                except Exception as e:
                    print(f"\nWarning: Failed to tokenize text: {e}")
                    continue
                    
            pbar.update(1)
        
        # Save final chunk if there's remaining data
        if chunk_buffer:
            is_train = (chunk_num < int(0.9 * total_rows))
            save_pattern = train_pattern if is_train else val_pattern
            save_chunk(chunk_buffer, save_pattern, chunk_num)
            chunk_num += 1
    
    # Save metadata
    metadata = {
        'num_chunks': chunk_num,
        'total_tokens': total_tokens,
        'chunk_size': chunk_size,
        'train_pattern': train_pattern,
        'val_pattern': val_pattern,
        'train_chunks': int(0.9 * chunk_num),
        'val_chunks': chunk_num - int(0.9 * chunk_num)
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nProcessed {total_tokens:,} tokens into {chunk_num} chunks")
    print(f"Training chunks: {metadata['train_chunks']}")
    print(f"Validation chunks: {metadata['val_chunks']}")
    return metadata

def preprocess_dataset():
    """Function to preprocess and cache the dataset"""
    print(f"Starting dataset preprocessing from: {dataset}")
    print(f"Using chunk size: {chunk_size}")
    print(f"Cache directory: {data_cache_dir}")
    
    try:
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"Dataset file not found: {dataset}")
            
        metadata = process_and_save_data(dataset, data_cache_dir, chunk_size)
        print("\nPreprocessing completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False

if __name__ == "__main__":
    # Run preprocessing when script is executed directly
    success = preprocess_dataset()
    if success:
        print("Dataset is ready for training!")
    else:
        print("Failed to preprocess dataset. Please check the errors above.")

class ChunkedDataset:
    def __init__(self, pattern, num_chunks, max_seq_len, chunk_size):
        self.pattern = pattern
        self.num_chunks = num_chunks
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.current_chunk_num = -1
        self.memory_manager = MemoryManager()
        self.start_token = START_TOKEN
        self.end_token = END_TOKEN
        self.pad_token = PAD_TOKEN
        self.vocab_size = encoder.n_vocab  # Add vocab_size for validation
    
    def load_random_chunk(self):
        """Load a random chunk from disk"""
        try:
            chunk_num = torch.randint(0, self.num_chunks, (1,)).item()
            self.current_chunk = load_chunk(self.pattern, chunk_num)
            self.current_chunk_num = chunk_num
        except Exception as e:
            print(f"Error loading chunk {chunk_num}: {e}")
            # Create an emergency chunk if loading fails
            self.current_chunk = [self.pad_token] * (self.max_seq_len * 2)
            self.current_chunk_num = -1
    
    def get_batch(self, batch_size, device):
        if self.current_chunk is None or len(self.current_chunk) < self.max_seq_len:
            self.load_random_chunk()
            
        max_start_idx = len(self.current_chunk) - self.max_seq_len - 1
        if max_start_idx <= 0:
            return self.get_batch(batch_size, device)
        
        try:
            ix = torch.randint(0, max_start_idx, (batch_size,))
            x = torch.stack([torch.tensor(
                self._validate_tokens(self.current_chunk[i:i+self.max_seq_len])
            ) for i in ix])
            y = torch.stack([torch.tensor(
                self._validate_tokens(self.current_chunk[i+1:i+self.max_seq_len+1])
            ) for i in ix])
            
            if device != 'cpu':
                x, y = x.to(device), y.to(device)
            return x, y
        except Exception as e:
            print(f"Error in get_batch: {e}")
            # Return a safe fallback batch
            return self._get_fallback_batch(batch_size, device)
    
    def _validate_tokens(self, tokens):
        """Ensure all tokens are within vocabulary range"""
        valid_tokens = []
        for token in tokens:
            if 0 <= token < self.vocab_size:
                valid_tokens.append(token)
            else:
                valid_tokens.append(self.pad_token)  # Replace invalid tokens with PAD
        return valid_tokens
    
    def _get_fallback_batch(self, batch_size, device):
        """Return a safe fallback batch when errors occur"""
        x = torch.full((batch_size, self.max_seq_len), self.pad_token, dtype=torch.long)
        y = torch.full((batch_size, self.max_seq_len), self.pad_token, dtype=torch.long)
        return x.to(device), y.to(device)

# Global datasets dictionary
datasets = None

def init_datasets(max_seq_len):
    """Initialize datasets with chunked data loading"""
    metadata = process_and_save_data(dataset, data_cache_dir, chunk_size)
    return {
        'train': ChunkedDataset(
            metadata['train_pattern'],
            metadata['train_chunks'],
            max_seq_len,
            metadata['chunk_size']
        ),
        'val': ChunkedDataset(
            metadata['val_pattern'],
            metadata['val_chunks'],
            max_seq_len,
            metadata['chunk_size']
        )
    }

def get_batch(split, batch_size, max_seq_len, device):
    """Get a batch of data from the specified split"""
    global datasets
    if datasets is None:
        print("Initializing datasets...")
        datasets = init_datasets(max_seq_len)
    return datasets[split].get_batch(batch_size, device)
