import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset, DataLoader
from model import XVADModel
from dataset import AVADataset, KaggleVADDataset
import os
import wandb
import argparse

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    else:
        print("Not using distributed mode.")
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_dataloader(dataset, batch_size, world_size, rank):
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument(
        "--dataset",
        type=str,
        default="kaggle",
        choices=["kaggle", "ava", "kaggle_ava"],
    )
    parser.add_argument(
        "--kaggle_label",
        type=str,
        default="/hpc_stor03/public/shared/data/mml/kaggle/vad/data/train_label.txt",
    )
    parser.add_argument(
        "--kaggle_audio",
        type=str,
        default="/hpc_stor03/public/shared/data/mml/kaggle/vad/wavs",  
    )
    parser.add_argument(
        "--ava_label",
        type=str,
        default="/hpc_stor03/public/shared/data/mml/AVAVD/annotations/labs",
    )
    parser.add_argument(
        "--ava_audio",
        type=str,
        default="/hpc_stor03/public/shared/data/mml/AVAVD/audios",
    )
    args = parser.parse_args()

    global_rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    is_master = (global_rank == 0)
    
    if is_master:
        print(f"Starting training on {world_size} GPUs.")
        wandb.init(
            project="xvad-training",
            config={
                "learning_rate": args.lr,
                "architecture": "CRNN",
                "dataset": (
                    "Kaggle-VAD" if args.dataset == "kaggle"
                    else ("AVA-Speech" if args.dataset == "ava" 
                    else "Kaggle+AVA")
                ),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "world_size": world_size
            }
        )
    
    model = XVADModel().to(device)
    
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    if args.dataset == "ava":
        lab_dir = args.ava_label
        audio_dir = args.ava_audio
        if not os.path.isdir(lab_dir):
            if is_master:
                print(f"Error: AVA lab dir {lab_dir} not found.")
            cleanup_distributed()
            return
        if not os.path.isdir(audio_dir):
            if is_master:
                print(f"Error: AVA audio dir {audio_dir} not found.")
            cleanup_distributed()
            return
        if is_master:
            print("Running sanity check on AVA-Speech dataset (Rank 0)...")
            import time
            t0 = time.time()
            try:
                test_dataset = AVADataset(lab_dir, audio_dir)
                if len(test_dataset) > 0:
                    _ = test_dataset[0]
                    print(f"Sanity check passed. Single item load time: {time.time()-t0:.4f}s")
                else:
                    print("Dataset is empty.")
            except Exception as e:
                print(f"Sanity check FAILED: {e}")
                wandb.finish()
                cleanup_distributed()
                return
        train_dataset = AVADataset(lab_dir, audio_dir)
        if len(train_dataset) == 0:
            if is_master:
                print("Error: AVA dataset is empty. Please check your lab and audio paths.")
            wandb.finish()
            cleanup_distributed()
            return
    elif args.dataset == "kaggle":
        if not os.path.exists(args.kaggle_label):
            if is_master:
                print(f"Error: Kaggle label file {args.kaggle_label} not found.")
            cleanup_distributed()
            return
        if not os.path.exists(args.kaggle_audio):
            if is_master:
                print(f"Error: Kaggle audio dir {args.kaggle_audio} not found.")
            cleanup_distributed()
            return
        if is_master:
            print("Running sanity check on Kaggle VAD dataset (Rank 0)...")
            import time
            t0 = time.time()
            try:
                test_dataset = KaggleVADDataset(args.kaggle_label, args.kaggle_audio)
                if len(test_dataset) > 0:
                    _ = test_dataset[0]
                    print(f"Sanity check passed. Single item load time: {time.time()-t0:.4f}s")
                else:
                    print("Kaggle dataset is empty. Please check your audio directory and labels.")
            except Exception as e:
                print(f"Kaggle sanity check FAILED: {e}")
                wandb.finish()
                cleanup_distributed()
                return
        train_dataset = KaggleVADDataset(args.kaggle_label, args.kaggle_audio)
        if len(train_dataset) == 0:
            if is_master:
                print("Error: Kaggle dataset is empty. Please check your audio directory and labels.")
            wandb.finish()
            cleanup_distributed()
            return
    else:
        lab_dir = args.ava_label
        audio_dir = args.ava_audio
        if not os.path.isdir(lab_dir):
            if is_master:
                print(f"Error: AVA lab dir {lab_dir} not found.")
            cleanup_distributed()
            return
        if not os.path.isdir(audio_dir):
            if is_master:
                print(f"Error: AVA audio dir {audio_dir} not found.")
            cleanup_distributed()
            return
        if not os.path.exists(args.kaggle_label):
            if is_master:
                print(f"Error: Kaggle label file {args.kaggle_label} not found.")
            cleanup_distributed()
            return
        if not os.path.exists(args.kaggle_audio):
            if is_master:
                print(f"Error: Kaggle audio dir {args.kaggle_audio} not found.")
            cleanup_distributed()
            return
        if is_master:
            print("Running sanity check on Kaggle+AVA dataset (Rank 0)...")
            import time
            t0 = time.time()
            try:
                kaggle_ds = KaggleVADDataset(args.kaggle_label, args.kaggle_audio)
                ava_ds = AVADataset(lab_dir, audio_dir)
                if len(kaggle_ds) > 0 and len(ava_ds) > 0:
                    _ = kaggle_ds[0]
                    _ = ava_ds[0]
                    print(f"Sanity check passed. Single item load time: {time.time()-t0:.4f}s")
                else:
                    print("Kaggle or AVA dataset is empty. Please check your data.")
            except Exception as e:
                print(f"Kaggle+AVA sanity check FAILED: {e}")
                wandb.finish()
                cleanup_distributed()
                return
        kaggle_ds = KaggleVADDataset(args.kaggle_label, args.kaggle_audio)
        ava_ds = AVADataset(lab_dir, audio_dir)
        if len(kaggle_ds) == 0 or len(ava_ds) == 0:
            if is_master:
                print("Error: Kaggle or AVA dataset is empty. Please check your data.")
            wandb.finish()
            cleanup_distributed()
            return
        train_dataset = ConcatDataset([kaggle_ds, ava_ds])

    train_loader, train_sampler = get_dataloader(train_dataset, args.batch_size, world_size, global_rank)
    
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
        scheduler.step()
        
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
