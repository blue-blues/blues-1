import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

# Number of workers for parallel processing
num_proc = 8
num_proc_load_dataset = num_proc

# Initialize tiktoken o200k_base tokenizer
enc = tiktoken.get_encoding("o200k_base")


def process_example(example):
    """Tokenize text and add end-of-text token"""
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)  # Add end-of-text token
    return {'ids': ids, 'len': len(ids)}


def clean_dataset(dataset):
    """Remove empty or excessively long sequences"""
    dataset = dataset.filter(lambda example: example['len'] > 10 and example['len'] < 1024)
    return dataset


if __name__ == '__main__':
    # Load BookCorpus dataset
    dataset = load_dataset("Shahzebbb/redpajama_100gb_processed", num_proc=num_proc_load_dataset)

    # Split dataset into training and validation sets
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename test split to validation

    # Tokenize and clean dataset
    tokenized_dataset = split_dataset.map(
        process_example,
        remove_columns=['text'],
        desc="Tokenizing and cleaning dataset",
        num_proc=num_proc,
    )
    cleaned_dataset = tokenized_dataset.map(
        clean_dataset,
        desc="Cleaning dataset",
        num_proc=num_proc,
    )

    # Save cleaned dataset to disk
    cleaned_dataset.save_to_disk("cleaned_corpus")


**Changes Made:**

1.  Introduced `process_example` function for tokenization.
2.  Added `clean_dataset` function to remove empty or excessively long sequences.
3.  Modified `process` function to append end-of-text token.
4.  Used `Dataset` from `datasets` library for better handling.
5.  Saved cleaned dataset to disk using `save_to_disk` method.


**Enhancements:**

1.  Parallel processing using `num_proc` workers.
2.  Tokenization and cleaning performed in separate steps.
3.  Dataset saved to disk for future use.


**Notes:**

1.  Adjust `num_proc` and `num_proc_load_dataset` according to your system's capabilities.
2.  Modify `clean_dataset` function to suit your specific cleaning requirements.
