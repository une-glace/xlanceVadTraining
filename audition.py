import torch
import torchaudio
from dataset import SyntheticVADDataset
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Audition tool for VAD dataset")
    parser.add_argument("--speech_scp", type=str, default="WenetSpeech_16k/wav.scp")
    parser.add_argument("--noise_scp", type=str, default="musan_16k/wav.scp")
    parser.add_argument("--output", type=str, default="audition_sample.wav")
    parser.add_argument("--count", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()

    if not os.path.exists(args.speech_scp) or not os.path.exists(args.noise_scp):
        print(f"Error: SCP files not found. Looked for {args.speech_scp} and {args.noise_scp}")
        print("Run prepare_list.py first.")
        return

    print("Initializing dataset...")
    # verbose=False because get_audio_sample has its own print
    dataset = SyntheticVADDataset(args.speech_scp, args.noise_scp, verbose=False)
    
    for i in range(args.count):
        print(f"\nGenerating sample {i+1}/{args.count}...")
        mixed, snr = dataset.get_audio_sample(i)
        
        if mixed is not None:
            out_name = args.output
            if args.count > 1:
                base, ext = os.path.splitext(out_name)
                out_name = f"{base}_{i+1}{ext}"
            
            torchaudio.save(out_name, mixed, dataset.sample_rate)
            print(f"Saved to {out_name}")
        else:
            print("Failed to generate sample (loading error).")

if __name__ == "__main__":
    main()