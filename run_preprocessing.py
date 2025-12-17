import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random
import multiprocessing

# ================= Configuration =================
# 配置你的源目录和目标目录
DATASETS = [
    {
        "name": "WenetSpeech",
        "src": "/hpc_stor03/public/shared/data/asr/rawdata/WenetSpeech/data/audio/train/",
        "dst": "/hpc_stor03/sjtu_home/qingya.li/xlanceVadTraining/WenetSpeech_16k",
        "ext": ["opus", "wav", "flac", "mp3"],
        "max_files": 1000  # <--- 限制 1000 个文件 (约 10-20GB)，对于 VAD 训练完全足够
    },
    {
        "name": "MUSAN",
        "src": "/hpc_stor03/public/shared/data/raa/musan/noise/",
        "dst": "/hpc_stor03/sjtu_home/qingya.li/xlanceVadTraining/musan_16k",
        "ext": ["wav"],
        "max_files": None  # MUSAN 很小 (6GB)，全部保留
    }
]

TARGET_SR = 16000
# 自动检测 CPU 核数，保留 4 个给系统
NUM_JOBS = max(1, multiprocessing.cpu_count() - 4)
# =================================================

def resample_worker(args):
    input_path, output_path, target_sr = args
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(input_path)
        
        # Resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, waveform, target_sr, bits_per_sample=16)
        return True
    except Exception as e:
        return False

def process_dataset(config):
    print(f"\n[Processing {config['name']}]")
    print(f"Source: {config['src']}")
    print(f"Target: {config['dst']}")
    
    files = []
    print("Scanning files...")
    for root, dirs, filenames in os.walk(config['src']):
        for f in filenames:
            ext = f.split('.')[-1].lower()
            if ext in config['ext']:
                files.append(os.path.join(root, f))
                
    print(f"Found {len(files)} files.")
    
    # Shuffle and Limit
    random.shuffle(files)
    if config.get('max_files') and len(files) > config['max_files']:
        print(f"Limiting to random {config['max_files']} files to save space...")
        files = files[:config['max_files']]
    
    tasks = []
    for f in files:
        rel_path = os.path.relpath(f, config['src'])
        out_path = os.path.join(config['dst'], rel_path)
        out_path = os.path.splitext(out_path)[0] + ".wav"
        tasks.append((f, out_path, TARGET_SR))
        
    print(f"Resampling with {NUM_JOBS} jobs...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=NUM_JOBS) as executor:
        results = list(tqdm(executor.map(resample_worker, tasks), total=len(tasks)))
        success_count = sum(results)
        
    print(f"Completed {config['name']}: {success_count}/{len(files)} files.")
    
    # Generate SCP
    scp_path = os.path.join(config['dst'], "wav.scp")
    with open(scp_path, 'w') as f:
        for task, success in zip(tasks, results):
            if success:
                f.write(f"{task[1]}\n")
    print(f"Generated list: {scp_path}")
    return scp_path

def main():
    print(f"Detected {multiprocessing.cpu_count()} CPU cores. Using {NUM_JOBS} parallel jobs.")
    
    scp_files = {}
    
    for config in DATASETS:
        scp_path = process_dataset(config)
        scp_files[config['name']] = scp_path
        
    print("\n" + "="*50)
    print("All Preprocessing Done!")
    print(f"Speech SCP: {scp_files['WenetSpeech']}")
    print(f"Noise SCP:  {scp_files['MUSAN']}")
    print("="*50)
    print("You can now start training with:")
    print(f"python train_vad.py --batch_size 128")
    print("(Make sure to update dataset.py or prepare_list.py to point to these new scp files)")

if __name__ == "__main__":
    main()
