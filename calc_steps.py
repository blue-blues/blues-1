import pandas as pd
from setting import dataset, batch_size

def calculate_total_steps(epochs=1):
    total_samples = 0
    for chunk in pd.read_csv(dataset, chunksize=batch_size):
        total_samples += len(chunk)
    steps_per_epoch = total_samples / batch_size
    total_steps = steps_per_epoch * epochs
    print(f"Total samples: {total_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps for {epochs} epochs: {total_steps}")
    return total_steps

if __name__ == "__main__":
    for epochs in [1, 3, 5]:
        print(f"\nCalculating for {epochs} epochs:")
        calculate_total_steps(epochs)
