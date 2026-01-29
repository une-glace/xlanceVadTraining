import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from model import XVADModel
from dataset import KaggleVADDataset


def average_precision(scores, labels):
    if labels.sum() == 0:
        return 0.0
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]
    cum_tp = torch.cumsum(sorted_labels, dim=0)
    positions = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    precision = cum_tp.float() / positions
    ap = (precision * sorted_labels.float()).sum() / sorted_labels.sum().float()
    return ap.item()


def evaluate_thresholds(scores, labels, thresholds):
    results = []
    for t in thresholds:
        preds = (scores >= t).int()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        results.append((t, f1, precision, recall))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/xvad_best.pth")
    parser.add_argument("--label", type=str, default="kaggle/vad/data/dev_label.txt")
    parser.add_argument("--audio", type=str, default="kaggle/vad/wavs")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KaggleVADDataset(args.label, args.audio)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = XVADModel().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = BCELoss()

    total_loss = 0.0
    num_batches = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1
            all_scores.append(outputs.detach().cpu().view(-1))
            all_labels.append(labels.detach().cpu().view(-1))

    if num_batches == 0:
        print("Dev dataset is empty.")
        return

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels).int()

    avg_loss = total_loss / num_batches
    ap = average_precision(scores, labels)

    thresholds = [i / 20.0 for i in range(1, 20)]
    results = evaluate_thresholds(scores, labels, thresholds)
    best = max(results, key=lambda x: x[1])

    print("Dev avg_loss:", avg_loss)
    print("Dev AP:", ap)
    print("Best threshold:", best[0])
    print("Best F1:", best[1])
    print("Best precision:", best[2])
    print("Best recall:", best[3])
    print("All thresholds (th, F1, precision, recall):")
    for t, f1, p, r in results:
        print(f"{t:.3f}\t{f1:.4f}\t{p:.4f}\t{r:.4f}")


if __name__ == "__main__":
    main()
