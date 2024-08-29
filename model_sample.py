import torch
from a import GPT, GPTConfig  # Assuming these are defined in your model script
device = torch.device('cpu')  # Set device to CPU

def evaluate_model(model, val_loader, device, num_examples=1000):
    model.eval()
    num_correct = 0
    num_total = 0

    with torch.no_grad():  # Ensure no gradients are computed
        for _ in range(num_examples):
            x, y = val_loader.next_batch()  # Assuming this returns data and labels
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x)
            
            # Compute the predictions
            pred = logits.argmax(dim=-1)
            num_correct += (pred == y).sum().item()
            num_total += y.numel()

    accuracy = num_correct / num_total
    return accuracy

def main():
    device = torch.device('cpu')  # Set device to CPU

    config = GPTConfig(vocab_size=199998)
    model = GPT(config)
    model.load_state_dict(torch.load('/root/blues-1/log/model_05000.pt', map_location=device, strict=False))
    model.to(device)  # Ensure model is on CPU
    
    # Initialize your DataLoader here
    val_loader = DataLoaderLite(B=4, T=1024, process_rank=0, num_processes=1, split="val")

    # Evaluate the model
    accuracy = evaluate_model(model, val_loader, device)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
