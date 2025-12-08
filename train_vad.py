import torch
import torch.nn as nn
import torch.optim as optim
from model import XVADModel
from dataset import get_dataloader
import os

def train():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    
    # 2. Model, Loss, Optimizer
    model = XVADModel().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy for 0/1 classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Data
    train_loader = get_dataloader(batch_size=BATCH_SIZE)
    
    # 4. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device) # [B, 80, 200]
            labels = labels.to(device)     # [B, 100, 1]
            
            # Forward pass
            # Note: For simple batch training, we don't pass hidden state (it defaults to 0)
            # In advanced streaming training (Truncated BPTT), you would manage hidden states manually.
            outputs, _ = model(features)
            
            # Loss calculation
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Complete. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/xvad_epoch_{epoch+1}.pth")

    print("Training finished!")

if __name__ == "__main__":
    train()
