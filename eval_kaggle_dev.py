import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from model import XVADModel
from dataset import KaggleVADDataset

checkpoint = "checkpoints/xvad_epoch_1.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = KaggleVADDataset(
    "kaggle/vad/data/dev_label.txt",
    "kaggle/vad/wavs",
)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

model = XVADModel().to(device)                              
state_dict = torch.load(checkpoint, map_location=device)
model.load_state_dict(state_dict)
model.eval()

criterion = BCELoss()

total_loss = 0.0
num_batches = 0
total_frames = 0
correct_frames = 0

with torch.no_grad():
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs, _ = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        num_batches += 1

        preds = (outputs >= 0.5).float()
        correct_frames += (preds == labels).sum().item()
        total_frames += labels.numel()

avg_loss = total_loss / num_batches
acc = correct_frames / total_frames

print("Dev avg_loss:", avg_loss)
print("Dev frame_acc:", acc)