import argparse
import os
import csv
import torch
import torchaudio
from model import XVADModel


def load_model(checkpoint, device):
    model = XVADModel().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_audio_files(audio_dir):
    paths = []
    for root, _, files in os.walk(audio_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(".wav") or lower.endswith(".flac"):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def build_mel_transform(sample_rate=16000):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
    )


def infer_on_file(path, model, mel_transform, device, sample_rate=16000):
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    feature = mel_transform(waveform).squeeze(0)
    if feature.shape[1] % 2 != 0:
        feature = feature[:, :-1]
    feat_frames = feature.shape[1]
    x = feature.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs, _ = model(x)
    probs = outputs.squeeze(0).squeeze(-1).cpu()
    out_frames = probs.shape[0]
    prob_10ms = torch.zeros(feat_frames)
    for j in range(feat_frames):
        idx20 = min(j // 2, out_frames - 1)
        prob_10ms[j] = probs[idx20]
    return prob_10ms.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument(
        "--ref_csv",
        type=str,
        default="kaggle/vad/data/test_ce.csv",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.checkpoint, device)
    mel_transform = build_mel_transform()

    audio_files = get_audio_files(args.audio_dir)
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        return

    pred_probs = {}
    for path in audio_files:
        base = os.path.splitext(os.path.basename(path))[0]
        probs = infer_on_file(path, model, mel_transform, device)
        for idx, p in enumerate(probs):
            uttid = f"{base}-{idx}"
            pred_probs[uttid] = float(p)

    if not os.path.exists(args.ref_csv):
        print(f"Reference CSV {args.ref_csv} not found.")
        return

    total = 0
    correct = 0
    with open(args.ref_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uttid = row.get("uttid")
            label_str = row.get("label")
            if uttid is None or label_str is None:
                continue
            if uttid not in pred_probs:
                continue
            try:
                ref_val = float(label_str)
            except ValueError:
                continue
            gt = 1 if ref_val >= 0.5 else 0
            pred = 1 if pred_probs[uttid] >= args.threshold else 0
            total += 1
            if gt == pred:
                correct += 1

    if total == 0:
        print("No matching frames between predictions and reference CSV.")
        return

    acc = correct / total
    print(f"Accuracy compared to {args.ref_csv}: {acc:.6f} (correct={correct}, total={total})")


if __name__ == "__main__":
    main()

