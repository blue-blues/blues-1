import os
import numpy as np
import tiktoken
import torch
from pathlib import Path
import csv
import pickle
from tqdm import tqdm
from setting import dataset, data_cache_dir, chunk_size

# Initialize tiktoken encoder
encoder = tiktoken.get_encoding("o200k_base")

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
                tokens = encoder.encode(row['Text'])
                chunk_buffer.extend(tokens)
                total_tokens += len(tokens)
                
                # Save chunk when buffer is full
                if len(chunk_buffer) >= chunk_size * 100:  # Multiply by 100 for token-based chunks
                    is_train = (chunk_num < int(0.9 * total_rows))
                    save_pattern = train_pattern if is_train else val_pattern
                    save_chunk(chunk_buffer, save_pattern, chunk_num)
                    chunk_buffer = []
                    chunk_num += 1
            
            pbar.update(1)
        
        # Save final chunk if there's data remaining
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
    
    print(f"Processed {total_tokens} tokens into {chunk_num} chunks")
    return metadata

class ChunkedDataset:
    def __init__(self, pattern, num_chunks, max_seq_len, chunk_size):
        self.pattern = pattern
        self.num_chunks = num_chunks
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.current_chunk_num = -1
        print(f"Initialized dataset with {num_chunks} chunks, pattern: {pattern}")
    
    def load_random_chunk(self):
        chunk_num = torch.randint(0, self.num_chunks, (1,)).item()
        if chunk_num != self.current_chunk_num:
            try:
                self.current_chunk = load_chunk(self.pattern, chunk_num)
                self.current_chunk_num = chunk_num
            except Exception as e:
                print(f"Error loading chunk {chunk_num}: {e}")
                raise
    
    def get_batch(self, batch_size, device):
        self.load_random_chunk()
        max_start_idx = len(self.current_chunk) - self.max_seq_len - 1
        if max_start_idx <= 0:
            return self.get_batch(batch_size, device)  # Recursively try another chunk
        
        ix = torch.randint(0, max_start_idx, (batch_size,))
        x = torch.stack([torch.tensor(self.current_chunk[i:i+self.max_seq_len]) for i in ix])
        y = torch.stack([torch.tensor(self.current_chunk[i+1:i+self.max_seq_len+1]) for i in ix])
        return x.to(device), y.to(device)

def init_datasets(max_seq_len):
    """Initialize datasets with chunked data loading"""
    metadata = process_and_save_data(dataset, data_cache_dir, chunk_size)
    print("Initializing datasets from processed chunks:")
    print(f"Train chunks: {metadata['train_chunks']}")
    print(f"Val chunks: {metadata['val_chunks']}")
    print(f"Total tokens: {metadata['total_tokens']}")
    
    train_dataset = ChunkedDataset(
        metadata['train_pattern'], 
        metadata['train_chunks'], 
        max_seq_len, 
        metadata['chunk_size']
    )
    val_dataset = ChunkedDataset(
        metadata['val_pattern'], 
        metadata['val_chunks'], 
        max_seq_len, 
        metadata['chunk_size']
    )
    
    # Verify data loading by testing a batch
    test_batch = train_dataset.get_batch(1, 'cpu')
    print(f"Verified data loading. Sample sequence length: {test_batch[0].shape[1]}")
    
    return {'train': train_dataset, 'val': val_dataset}

def get_batch(split, batch_size, max_seq_len, device):
    """Get a batch of data from the specified split"""
    global datasets
    if 'datasets' not in globals():
        print("First time data access - initializing datasets...")
        datasets = init_datasets(max_seq_len)
        if not os.path.exists(data_cache_dir):
            raise RuntimeError(f"Data cache directory {data_cache_dir} not found!")
    return datasets[split].get_batch(batch_size, device)
