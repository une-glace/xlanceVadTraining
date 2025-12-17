import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model import XVADModel
from dataset import SyntheticVADDataset
from torch.utils.data import DataLoader
import os
import wandb
import argparse

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    else:
        # Fallback to single GPU
        print("Not using distributed mode.")
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_dataloader(speech_scp, noise_scp, batch_size, world_size, rank, verbose=False):
    dataset = SyntheticVADDataset(speech_scp, noise_scp, verbose=verbose)
    
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(sampler is None), 
        num_workers=4, 
        sampler=sampler,
        pin_memory=True
    )
    return dataloader, sampler

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64) # Increased batch size for multi-gpu
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose dataset logging")
    args = parser.parse_args()

    # 1. Setup Distributed
    global_rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    is_master = (global_rank == 0)
    
    if is_master:
        print(f"Starting training on {world_size} GPUs.")
        # Initialize wandb only on master process
        wandb.init(
            project="xvad-training",
            config={
                "learning_rate": args.lr,
                "architecture": "CRNN",
                "dataset": "WenetSpeech+MUSAN",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "world_size": world_size
            }
        )
    
    # 2. Model, Loss, Optimizer
    model = XVADModel().to(device)
    
    # Convert BatchNorm to SyncBatchNorm for better statistics across GPUs
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Data
    speech_scp = "speech.scp"
    noise_scp = "noise.scp"
    
    if not os.path.exists(speech_scp) or not os.path.exists(noise_scp):
        if is_master:
            print(f"Error: {speech_scp} or {noise_scp} not found.")
            print("Please run 'python prepare_list.py --speech_dir ... --noise_dir ...' first.")
            wandb.finish()
        cleanup_distributed()
        return

    # Sanity Check (Master Only)
    if is_master:
        print("Running sanity check on dataset (Rank 0)...")
        import time
        t0 = time.time()
        # Create a temporary dataset for check
        test_dataset = SyntheticVADDataset(speech_scp, noise_scp, verbose=True)
        try:
            _ = test_dataset[0]
            print(f"Sanity check passed. Single item load time: {time.time()-t0:.4f}s")
        except Exception as e:
            print(f"Sanity check FAILED: {e}")
            wandb.finish()
            cleanup_distributed()
            return

    # Only master gets verbose logs if requested
    # Default to False to keep terminal clean unless debugging
    train_loader, train_sampler = get_dataloader(speech_scp, noise_scp, args.batch_size, world_size, global_rank, verbose=(args.verbose and is_master))
    
    # 4. Training Loop
    if is_master:
        print("Starting training...")
        
    for epoch in range(args.epochs):
        if is_master:
            print(f"\n=== Starting Epoch {epoch+1}/{args.epochs} ===")
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Reduce loss for logging (optional, here we just log master's loss to save time)
            current_loss = loss.item()
            total_loss += current_loss
            
            if is_master:
                wandb.log({"batch_loss": current_loss})
                # Print every step for the first 10 steps to debug startup speed
                if batch_idx < 10 or batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}], Loss: {current_loss:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        
        if is_master:
            print(f"Epoch [{epoch+1}/{args.epochs}] Complete. Average Loss: {avg_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss
            })
            
            # Save checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            # Access underlying model in DDP
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, f"checkpoints/xvad_epoch_{epoch+1}.pth")

    if is_master:
        print("Training finished!")
        wandb.finish()
        
    cleanup_distributed()

if __name__ == "__main__":
    train()
