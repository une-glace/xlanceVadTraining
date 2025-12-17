import os
import glob
import argparse
from pathlib import Path
import random

def scan_files(directory, extensions=['.wav', '.flac', '.mp3', '.opus']):
    """Recursively scan for audio files."""
    audio_files = []
    directory = Path(directory)
    print(f"Scanning {directory}...")
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
                
    return audio_files

def write_scp(files, output_file):
    """Write file paths to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in files:
            f.write(f"{file_path}\n")
    print(f"Saved {len(files)} files to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare audio file lists for VAD training")
    parser.add_argument("--speech_dir", type=str, required=True, help="Path to speech dataset (e.g., WenetSpeech/data/audio)")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to noise dataset (e.g., MUSAN/noise)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save .scp files")
    
    args = parser.parse_args()
    
    # 1. Scan Speech Data
    # Note: For WenetSpeech, we might only want 'train_s' or 'train_m' for quick experiments
    # If the path is huge, this might take a while.
    speech_files = scan_files(args.speech_dir)
    if not speech_files:
        print(f"Warning: No audio files found in {args.speech_dir}")
    
    # 2. Scan Noise Data
    noise_files = scan_files(args.noise_dir)
    if not noise_files:
        print(f"Warning: No audio files found in {args.noise_dir}")
        
    # 3. Save
    os.makedirs(args.output_dir, exist_ok=True)
    write_scp(speech_files, os.path.join(args.output_dir, "speech.scp"))
    write_scp(noise_files, os.path.join(args.output_dir, "noise.scp"))
    
    print("Done! You can now use speech.scp and noise.scp for training.")

if __name__ == "__main__":
    main()
