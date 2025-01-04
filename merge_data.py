import os
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
from setting import data_cache_dir

class DatasetMerger:
    def __init__(self, output_dir=data_cache_dir):
        self.output_dir = output_dir
        self.merged_dir = os.path.join(output_dir, 'merged')
        os.makedirs(self.merged_dir, exist_ok=True)
        
    def merge_datasets(self, dataset_paths, weights=None):
        """
        Merge multiple preprocessed datasets
        
        Args:
            dataset_paths (list): List of paths to different dataset cache directories
            weights (list, optional): List of weights for each dataset
        """
        print("Starting dataset merge process...")
        
        # Validate and load all metadata
        all_metadata = []
        total_chunks = 0
        total_tokens = 0
        
        for path in dataset_paths:
            metadata_file = os.path.join(path, 'metadata.pkl')
            if not os.path.exists(metadata_file):
                raise ValueError(f"No metadata found in {path}")
                
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                all_metadata.append(metadata)
                total_chunks += metadata['num_chunks']
                total_tokens += metadata['total_tokens']
        
        # Initialize merged dataset structure
        train_chunks = []
        val_chunks = []
        chunk_counter = 0
        
        # Process each dataset
        for idx, (path, metadata) in enumerate(zip(dataset_paths, all_metadata)):
            weight = weights[idx] if weights else 1.0
            print(f"\nProcessing dataset {idx+1}/{len(dataset_paths)} (weight: {weight})")
            
            # Process training chunks
            for i in range(metadata['train_chunks']):
                src_file = metadata['train_pattern'].format(i)
                if os.path.exists(src_file):
                    with open(src_file, 'rb') as f:
                        chunk_data = pickle.load(f)
                        
                    # Apply weight by repeating data if weight > 1
                    if weight > 1:
                        chunk_data = chunk_data * int(weight)
                    elif weight < 1:
                        chunk_data = chunk_data[:int(len(chunk_data) * weight)]
                    
                    # Save merged chunk
                    dst_file = os.path.join(self.merged_dir, f'train_chunk_{chunk_counter}.pkl')
                    with open(dst_file, 'wb') as f:
                        pickle.dump(chunk_data, f)
                    train_chunks.append(chunk_counter)
                    chunk_counter += 1
            
            # Process validation chunks similarly
            for i in range(metadata['val_chunks']):
                src_file = metadata['val_pattern'].format(i)
                if os.path.exists(src_file):
                    dst_file = os.path.join(self.merged_dir, f'val_chunk_{chunk_counter}.pkl')
                    shutil.copy2(src_file, dst_file)
                    val_chunks.append(chunk_counter)
                    chunk_counter += 1
        
        # Create merged metadata
        merged_metadata = {
            'num_chunks': chunk_counter,
            'total_tokens': total_tokens,
            'chunk_size': all_metadata[0]['chunk_size'],
            'train_pattern': os.path.join(self.merged_dir, 'train_chunk_{}.pkl'),
            'val_pattern': os.path.join(self.merged_dir, 'val_chunk_{}.pkl'),
            'train_chunks': len(train_chunks),
            'val_chunks': len(val_chunks),
            'source_datasets': dataset_paths,
            'weights': weights if weights else [1.0] * len(dataset_paths)
        }
        
        # Save merged metadata
        with open(os.path.join(self.merged_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(merged_metadata, f)
        
        print(f"\nMerge completed:")
        print(f"Total chunks: {chunk_counter}")
        print(f"Training chunks: {len(train_chunks)}")
        print(f"Validation chunks: {len(val_chunks)}")
        print(f"Total tokens: {total_tokens}")
        
        return self.merged_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Merge multiple preprocessed datasets')
    parser.add_argument('--datasets', nargs='+', required=True, help='Paths to dataset directories')
    parser.add_argument('--weights', nargs='+', type=float, help='Weights for each dataset')
    args = parser.parse_args()
    
    if args.weights and len(args.weights) != len(args.datasets):
        raise ValueError("Number of weights must match number of datasets")
    
    merger = DatasetMerger()
    merged_dir = merger.merge_datasets(args.datasets, args.weights)
    print(f"\nMerged dataset saved to: {merged_dir}")
