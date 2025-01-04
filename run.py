import argparse
import os
from pathlib import Path

def setup_environment():
    """Create necessary directories and validate environment"""
    dirs = ['checkpoints', 'data_cache', 'logs', 'hf_cache']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Blues MoE Runner')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'generate', 'merge', 'validate'], required=True)
    parser.add_argument('--config', type=str, default='configs/default.json', help='Configuration file')
    parser.add_argument('--data_source', choices=['csv', 'huggingface'], default='csv')
    parser.add_argument('--datasets', nargs='+', help='Dataset paths for merging')
    parser.add_argument('--weights', nargs='+', type=float, help='Dataset weights for merging')
    
    args = parser.parse_args()
    setup_environment()

    if args.mode == 'preprocess':
        from data import preprocess_dataset
        preprocess_dataset()
    
    elif args.mode == 'train':
        from train import main as train_main
        train_main()
    
    elif args.mode == 'generate':
        from generate import main as generate_main
        generate_main()
    
    elif args.mode == 'merge':
        from merge_data import DatasetMerger
        merger = DatasetMerger()
        merger.merge_datasets(args.datasets, args.weights)

if __name__ == "__main__":
    main()
