import os
import torch
import pickle
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from setting import data_cache_dir, hf_dataset_config, chunk_size
from data import encoder, save_chunk, ChunkedDataset

class HFDatasetLoader:
    def __init__(self, 
                 dataset_name: str = hf_dataset_config['name'],
                 subset: Optional[str] = hf_dataset_config['subset'],
                 split: str = hf_dataset_config['split'],
                 text_column: str = hf_dataset_config['text_column'],
                 cache_dir: str = hf_dataset_config['cache_dir'],
                 data_dir: str = hf_dataset_config['data_dir'],
                 num_workers: int = os.cpu_count() or 1,
                 cache_mode: str = 'memory'):
        
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.cache_dir = cache_dir
        self.text_column = text_column
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.cache_mode = cache_mode
        self.metadata_file = os.path.join(data_cache_dir, f'{dataset_name.replace("/", "_")}_metadata.pkl')
        self.stats: Dict[str, Any] = {}
        
        # Create cache directories
        for directory in [cache_dir, data_dir, data_cache_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize cache
        self._init_cache()
        
    def _init_cache(self):
        """Initialize caching system"""
        self.cache = {}
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'rb') as f:
                    self.stats = pickle.load(f)
            except Exception as e:
                print(f"Failed to load metadata: {e}")
                self.stats = {}
                
    def _validate_data(self, data):
        """Validate dataset entries"""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        if self.text_column not in data:
            raise KeyError(f"Text column '{self.text_column}' not found")
            
    def process_chunk(self, texts):
        """Process a chunk of texts with error handling"""
        try:
            tokens = [encoder.encode(text) for text in texts]
            return tokens
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return None
            
    def load_dataset(self):
        """Load dataset with progress tracking and validation"""
        try:
            dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                cache_dir=self.cache_dir
            )
            
            # Compute dataset statistics
            self.stats.update({
                'total_samples': len(dataset),
                'avg_length': np.mean([len(x[self.text_column]) for x in dataset]),
                'vocab_size': len(encoder),
            })
            
            # Process in chunks with multiprocessing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
                futures = []
                
                for chunk in chunks:
                    texts = [x[self.text_column] for x in chunk]
                    futures.append(executor.submit(self.process_chunk, texts))
                
                # Track progress
                processed_chunks = []
                for future in tqdm(futures, desc="Processing chunks"):
                    result = future.result()
                    if result is not None:
                        processed_chunks.extend(result)
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.stats, f)
                
            return ChunkedDataset(processed_chunks, self.stats)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return self.stats

def prepare_hf_dataset(dataset_name=None, subset=None, split=None, text_column=None):
    """Convenience function to prepare a HuggingFace dataset"""
    loader = HFDatasetLoader(
        dataset_name=dataset_name or hf_dataset_config['name'],
        subset=subset or hf_dataset_config['subset'],
        split=split or hf_dataset_config['split'],
        text_column=text_column or hf_dataset_config['text_column']
    )
    return loader.load_dataset()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process HuggingFace dataset')
    parser.add_argument('--dataset', type=str, help='HuggingFace dataset name')
    parser.add_argument('--subset', type=str, help='Dataset subset')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--text_column', type=str, help='Text column name')
    parser.add_argument('--num_proc', type=int, default=16, help='Number of processes for processing')
    args = parser.parse_args()

    metadata = prepare_hf_dataset(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        text_column=args.text_column
    )
