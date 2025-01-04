import os
import torch
import pickle
from datasets import load_dataset
from tqdm import tqdm
from setting import data_cache_dir, hf_dataset_config, chunk_size
from data import encoder, save_chunk, ChunkedDataset

class HFDatasetLoader:
    def __init__(self, 
                 dataset_name=hf_dataset_config['name'],
                 subset=hf_dataset_config['subset'],
                 split=hf_dataset_config['split'],
                 text_column=hf_dataset_config['text_column'],
                 cache_dir=hf_dataset_config['cache_dir']):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.cache_dir = cache_dir
        self.text_column = text_column
        self.metadata_file = os.path.join(data_cache_dir, f'{dataset_name.replace("/", "_")}_metadata.pkl')
        
        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_cache_dir, exist_ok=True)

    def process_data(self, streaming=True, num_proc=4):
        """Process HuggingFace dataset with multi-processing support"""
        if os.path.exists(self.metadata_file):
            print(f"Found cached dataset: {self.dataset_name}")
            return self.load_metadata()

        print(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=streaming,
            num_proc=num_proc if not streaming else None,
            cache_dir=self.cache_dir
        )

        return self.chunk_and_save(dataset)

    def chunk_and_save(self, dataset):
        """Process dataset in chunks with progress tracking"""
        chunk_buffer = []
        total_tokens = 0
        chunk_num = 0
        train_chunks = []
        val_chunks = []

        # Initialize progress bar for streaming datasets
        pbar = tqdm(desc="Processing dataset")
        
        try:
            for item in dataset:
                if self.text_column in item:
                    tokens = encoder.encode(item[self.text_column])
                    chunk_buffer.extend(tokens)
                    total_tokens += len(tokens)
                    pbar.update(1)

                    if len(chunk_buffer) >= chunk_size * 100:
                        self._save_current_chunk(chunk_buffer, chunk_num, train_chunks, val_chunks)
                        chunk_buffer = []
                        chunk_num += 1
                        pbar.set_postfix({'chunks': chunk_num, 'tokens': total_tokens})

        except KeyboardInterrupt:
            print("\nProcessing interrupted. Saving progress...")
        finally:
            # Save any remaining data
            if chunk_buffer:
                self._save_current_chunk(chunk_buffer, chunk_num, train_chunks, val_chunks)
                chunk_num += 1

        metadata = self._create_metadata(chunk_num, total_tokens, train_chunks, val_chunks)
        self._save_metadata(metadata)
        
        print(f"\nProcessed {total_tokens} tokens into {chunk_num} chunks")
        print(f"Training chunks: {len(train_chunks)}")
        print(f"Validation chunks: {len(val_chunks)}")
        return metadata

    def _save_current_chunk(self, chunk_buffer, chunk_num, train_chunks, val_chunks):
        """Save current chunk and update tracking lists"""
        is_train = (chunk_num < int(0.9 * (chunk_num + 1)))
        pattern = os.path.join(data_cache_dir,
            f'{"train" if is_train else "val"}_chunk_{chunk_num}.pkl')
        save_chunk(chunk_buffer, pattern, chunk_num)
        
        if is_train:
            train_chunks.append(chunk_num)
        else:
            val_chunks.append(chunk_num)

    def _create_metadata(self, chunk_num, total_tokens, train_chunks, val_chunks):
        """Create metadata dictionary"""
        return {
            'dataset_name': self.dataset_name,
            'num_chunks': chunk_num,
            'total_tokens': total_tokens,
            'chunk_size': chunk_size,
            'train_pattern': os.path.join(data_cache_dir, 'train_chunk_{}.pkl'),
            'val_pattern': os.path.join(data_cache_dir, 'val_chunk_{}.pkl'),
            'train_chunks': len(train_chunks),
            'val_chunks': len(val_chunks),
            'train_chunk_ids': train_chunks,
            'val_chunk_ids': val_chunks
        }

    def _save_metadata(self, metadata):
        """Save metadata to file"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

    def load_metadata(self):
        """Load dataset metadata"""
        with open(self.metadata_file, 'rb') as f:
            return pickle.load(f)

def prepare_hf_dataset(dataset_name=None, subset=None, split=None, text_column=None):
    """Convenience function to prepare a HuggingFace dataset"""
    loader = HFDatasetLoader(
        dataset_name=dataset_name or hf_dataset_config['name'],
        subset=subset or hf_dataset_config['subset'],
        split=split or hf_dataset_config['split'],
        text_column=text_column or hf_dataset_config['text_column']
    )
    return loader.process_data(streaming=hf_dataset_config['streaming'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process HuggingFace dataset')
    parser.add_argument('--dataset', type=str, help='HuggingFace dataset name')
    parser.add_argument('--subset', type=str, help='Dataset subset')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--text_column', type=str, help='Text column name')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes for processing')
    args = parser.parse_args()

    metadata = prepare_hf_dataset(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        text_column=args.text_column
    )
