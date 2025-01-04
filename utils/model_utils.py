import torch

def estimate_loss(model, data_loader, device):
    """Estimate the loss of the model on a given dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss = model(inputs, targets)
            total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader)
