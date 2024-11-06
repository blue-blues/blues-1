import torch
import torch.nn as nn
from models import SparseMoELanguageModel
from data_utils import get_batch
from parameters import *

def train(model, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in range(eval_iters):
            X, Y = get_batch("train")
            optimizer.zero_grad()
            _, loss = model(X, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / eval_iters}")
        
        # Evaluate model
        model.eval()
        val_loss = estimate_loss()
        print(f"Validation Loss: {val_loss['val']}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

def main():
    model = SparseMoELanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train(model, optimizer, epochs=10)

if __name__ == "__main__":
    main()