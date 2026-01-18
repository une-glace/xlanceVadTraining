import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
import os
import math
import time
import logging
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class SyntheticVADDataset(Dataset):
    def __init__(self, speech_scp, noise_scp, sample_rate=16000, duration=3.0, epoch_len=10000, verbose=False):
        """
        Args:
            speech_scp (str): Path to speech file list.
            noise_scp (str): Path to noise file list.
            sample_rate (int): Target sample rate.
            duration (float): Target duration in seconds (chunk size).
            epoch_len (int): Virtual length of one epoch (since we sample randomly).
            verbose (bool): Whether to print detailed processing steps.
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.epoch_len = epoch_len
        self.verbose = verbose
        
        # Load file lists
        self.speech_files = self._load_list(speech_scp)
        self.noise_files = self._load_list(noise_scp)
        
        print(f"Loaded {len(self.speech_files)} speech files and {len(self.noise_files)} noise files.")
        
        # Feature Extractor: MelSpectrogram
        # Win length 25ms, Hop length 10ms -> 100 frames per second
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )

    def _load_list(self, path):
        if not path or not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _load_audio(self, path, target_chunk_len=0):
        """
        Efficiently load an audio file.
        If target_chunk_len > 0, it attempts to read only a random chunk of that length.
        """
        try:
            if target_chunk_len > 0:
                # 1. Get metadata only
                info = torchaudio.info(path)
                orig_sr = info.sample_rate
                orig_len = info.num_frames
                
                # Calculate target length in original sample rate
                target_len_orig = int(target_chunk_len * (orig_sr / self.sample_rate))
                
                if orig_len > target_len_orig:
                    # Random seek
                    start_frame = random.randint(0, orig_len - target_len_orig)
                    waveform, sr = torchaudio.load(path, frame_offset=start_frame, num_frames=target_len_orig)
                else:
                    # File too short, read all
                    waveform, sr = torchaudio.load(path)
            else:
                waveform, sr = torchaudio.load(path)
            
            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            return waveform
        except Exception as e:
            # print(f"Error loading {path}: {e}")
            return None

    def _get_random_chunk(self, waveform, target_len):
        """Extract a random chunk of target_len from waveform."""
        if waveform.shape[1] < target_len:
            # Pad if too short
            padding = target_len - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Random crop
            start = random.randint(0, waveform.shape[1] - target_len)
            return waveform[:, start:start+target_len]

    def __len__(self):
        return self.epoch_len

    def get_audio_sample(self, idx):
        """
        Returns the raw mixed audio waveform and label for auditioning purposes.
        """
        # 1. Select Noise
        noise_path = random.choice(self.noise_files)
        noise_wav = self._load_audio(noise_path, target_chunk_len=self.target_len)
        if noise_wav is None: return None, None
        noise_chunk = self._get_random_chunk(noise_wav, self.target_len)
        
        # 2. Select Speech
        speech_path = random.choice(self.speech_files)
        speech_wav = self._load_audio(speech_path, target_chunk_len=self.target_len)
        if speech_wav is None: return None, None
        
        speech_len = speech_wav.shape[1]
        if speech_len > self.target_len:
                speech_chunk = self._get_random_chunk(speech_wav, random.randint(int(self.sample_rate*0.5), self.target_len))
        else:
                speech_chunk = speech_wav
        
        speech_len_samples = speech_chunk.shape[1]
        
        # 3. Mix Strategy
        mix_type = random.random()
        
        if mix_type < 0.1:
            # === Pure Noise ===
            mixed = noise_chunk
            snr_db = -999
        
        elif mix_type < 0.2:
            # === Pure Speech ===
            mixed = torch.zeros(1, self.target_len)
            max_start = self.target_len - speech_len_samples
            if max_start < 0: max_start = 0
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + speech_len_samples
            mixed[:, start_idx:end_idx] = speech_chunk
            snr_db = 999
            
        else:
            # === Mixed ===
            snr_db = random.uniform(5, 20)
            speech_power = speech_chunk.norm(p=2)
            noise_power = noise_chunk.norm(p=2)
            
            if noise_power == 0:
                scale = 0
            else:
                scale = math.pow(10, -snr_db / 20) * (speech_power / noise_power)
            
            max_start = self.target_len - speech_len_samples
            if max_start < 0: max_start = 0
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + speech_len_samples
            
            mixed = noise_chunk.clone() * scale
            mixed[:, start_idx:end_idx] += speech_chunk
        
        print(f"Audition Sample: Speech={os.path.basename(speech_path)}, Noise={os.path.basename(noise_path)}, SNR={snr_db:.2f}dB")
        
        return mixed, snr_db

    def __getitem__(self, idx):
        # Retry logic in case of bad files
        for retry in range(5):
            try:
                t0 = time.time()
                
                # 1. Decide Mix Strategy first
                # 10% Pure Noise (No speech)
                # 10% Pure Speech (No noise)
                # 80% Mixed (SNR 5-20dB)
                mix_type = random.random()
                
                noise_chunk = None
                speech_chunk = None
                mixed = None
                start_idx = 0
                end_idx = 0
                snr_db = 0
                
                # === Case A: Pure Noise (10%) ===
                if mix_type < 0.1:
                    if self.noise_files:
                        noise_path = random.choice(self.noise_files)
                        noise_wav = self._load_audio(noise_path, target_chunk_len=self.target_len)
                        if noise_wav is None: continue
                        noise_chunk = self._get_random_chunk(noise_wav, self.target_len)
                        
                        mixed = noise_chunk
                        if self.verbose and retry == 0:
                            print(f"[Dataset] Pure Noise. {os.path.basename(noise_path)}")
                    else:
                        # Synthetic Noise
                        mixed = torch.randn(1, self.target_len) * 0.01
                        if self.verbose and retry == 0:
                            print(f"[Dataset] Pure Noise (Synthetic).")

                    start_idx = 0
                    end_idx = 0 # No speech
                    snr_db = -999

                # === Case B: Pure Speech (10%) ===
                elif mix_type < 0.2:
                    speech_path = random.choice(self.speech_files)
                    speech_wav = self._load_audio(speech_path, target_chunk_len=self.target_len)
                    if speech_wav is None: continue
                    
                    # Process speech chunk
                    speech_len = speech_wav.shape[1]
                    if speech_len > self.target_len:
                         speech_chunk = self._get_random_chunk(speech_wav, random.randint(int(self.sample_rate*0.5), self.target_len))
                    else:
                         speech_chunk = speech_wav
                    
                    speech_len_samples = speech_chunk.shape[1]
                    
                    # Create canvas
                    mixed = torch.zeros(1, self.target_len)
                    max_start = self.target_len - speech_len_samples
                    if max_start < 0: max_start = 0
                    start_idx = random.randint(0, max_start)
                    end_idx = start_idx + speech_len_samples
                    
                    mixed[:, start_idx:end_idx] = speech_chunk
                    snr_db = 999
                    
                    if self.verbose and retry == 0:
                        print(f"[Dataset] Pure Speech. {os.path.basename(speech_path)}")

                # === Case C: Mixed (80%) ===
                else:
                    # Speech
                    speech_path = random.choice(self.speech_files)
                    speech_wav = self._load_audio(speech_path, target_chunk_len=self.target_len)
                    if speech_wav is None: continue
                    
                    # Process speech chunk
                    speech_len = speech_wav.shape[1]
                    if speech_len > self.target_len:
                         speech_chunk = self._get_random_chunk(speech_wav, random.randint(int(self.sample_rate*0.5), self.target_len))
                    else:
                         speech_chunk = speech_wav
                    
                    speech_len_samples = speech_chunk.shape[1]
                    
                    # Noise (Conditional)
                    noise_chunk = None
                    if self.noise_files:
                        noise_path = random.choice(self.noise_files)
                        noise_wav = self._load_audio(noise_path, target_chunk_len=self.target_len)
                        if noise_wav is not None:
                            noise_chunk = self._get_random_chunk(noise_wav, self.target_len)
                    
                    # SNR and Mixing
                    snr_db = random.uniform(5, 20)
                    speech_power = speech_chunk.norm(p=2)
                    
                    max_start = self.target_len - speech_len_samples
                    if max_start < 0: max_start = 0
                    start_idx = random.randint(0, max_start)
                    end_idx = start_idx + speech_len_samples
                    
                    if noise_chunk is not None:
                        noise_power = noise_chunk.norm(p=2)
                        if noise_power == 0:
                            scale = 0
                        else:
                            scale = math.pow(10, -snr_db / 20) * (speech_power / noise_power)
                        
                        mixed = noise_chunk.clone() * scale
                        mixed[:, start_idx:end_idx] += speech_chunk
                        
                        if self.verbose and retry == 0:
                            print(f"[Dataset] Mixed SNR {snr_db:.1f}dB. S:{os.path.basename(speech_path)} N:{os.path.basename(noise_path)}")
                    else:
                        # No noise available
                        mixed = torch.zeros(1, self.target_len)
                        mixed[:, start_idx:end_idx] = speech_chunk
                        snr_db = 999
                        if self.verbose and retry == 0:
                            print(f"[Dataset] Mixed (No Added Noise). S:{os.path.basename(speech_path)}")

                # === Common Processing: Label & Feature ===
                
                # 4. Generate Label
                total_frames = int(self.target_len / 160) + 1 
                label_len = total_frames // 2 
                label = torch.zeros(label_len, 1)
                
                effective_stride = 320 # hop_length * 2
                
                # Only mark ones if we have speech (end_idx > 0)
                if end_idx > 0:
                    start_frame = int(start_idx / effective_stride)
                    end_frame = int(end_idx / effective_stride)
                    if end_frame >= label_len: end_frame = label_len
                    label[start_frame:end_frame] = 1.0
                
                # 5. Extract Features
                feature = self.mel_spectrogram(mixed).squeeze(0) 
                
                # Fix Alignment
                if feature.shape[1] % 2 != 0:
                    feature = feature[:, :-1]
                    
                curr_label_len = feature.shape[1] // 2
                if label.shape[0] != curr_label_len:
                    new_label = torch.zeros(curr_label_len, 1)
                    min_len = min(curr_label_len, label.shape[0])
                    new_label[:min_len] = label[:min_len]
                    label = new_label
                    
                return feature, label
            
            except Exception as e:
                if self.verbose:
                    print(f"[Dataset] Error processing idx {idx}: {e}")
                continue

        # Fallback if all fail
        return torch.randn(80, 300), torch.zeros(150, 1)


def parse_kaggle_vad_label(line, frame_size: float = 0.025, frame_shift: float = 0.01):
    frame2time = lambda n: n * frame_shift + frame_size / 2
    frames = []
    frame_n = 0
    for time_pairs in line.split():
        start, end = map(float, time_pairs.split(","))
        if end <= start:
            continue
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


class KaggleVADDataset(Dataset):
    def __init__(self, label_path, audio_dir, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.audio_dir = audio_dir
        self.frame_size = 0.025
        self.frame_shift = 0.01
        self.items = []
        self.path_index = self._build_path_index()
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) < 2:
                        continue
                    utt_id, segs = parts
                    audio_path = self._resolve_audio_path(utt_id)
                    if audio_path is None:
                        continue
                    labels = parse_kaggle_vad_label(
                        segs, frame_size=self.frame_size, frame_shift=self.frame_shift
                    )
                    self.items.append(
                        {
                            "utt": utt_id,
                            "path": audio_path,
                            "labels": labels,
                        }
                    )
        logger.info(
            f"[KaggleVADDataset] Loaded {len(self.items)} items from audio_dir={self.audio_dir}"
        )
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )

    def _build_path_index(self):
        index = {}
        if os.path.exists(self.audio_dir):
            for root, _, files in os.walk(self.audio_dir):
                for name in files:
                    lower = name.lower()
                    if lower.endswith(".wav") or lower.endswith(".flac"):
                        key, _ = os.path.splitext(name)
                        full_path = os.path.join(root, name)
                        index[key] = full_path
        return index

    def _resolve_audio_path(self, utt_id):
        return self.path_index.get(utt_id)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            item = self.items[idx]
            path = item["path"]
            labels_full = item["labels"]
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                )
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            total_len = waveform.shape[1]
            if total_len >= self.target_len:
                max_start = total_len - self.target_len
                start_sample = random.randint(0, max_start)
                end_sample = start_sample + self.target_len
                segment = waveform[:, start_sample:end_sample]
            else:
                segment = torch.zeros(1, self.target_len)
                segment[:, :total_len] = waveform
                start_sample = 0
                end_sample = total_len
            feature = self.mel_spectrogram(segment).squeeze(0)
            if feature.shape[1] % 2 != 0:
                feature = feature[:, :-1]
            feat_frames = feature.shape[1]
            frame_shift_samples = int(self.sample_rate * self.frame_shift)
            start_frame = start_sample // frame_shift_samples
            end_frame = start_frame + feat_frames
            label_frames = labels_full[start_frame:end_frame]
            if len(label_frames) < feat_frames:
                pad_len = feat_frames - len(label_frames)
                label_frames = label_frames + [0] * pad_len
            else:
                label_frames = label_frames[:feat_frames]
            out_len = feat_frames // 2
            label = torch.zeros(out_len, 1)
            for i in range(out_len):
                a = label_frames[2 * i]
                b = 0
                if 2 * i + 1 < feat_frames:
                    b = label_frames[2 * i + 1]
                if a or b:
                    label[i, 0] = 1.0
            return feature, label
        except Exception:
            return torch.randn(80, 300), torch.zeros(150, 1)


class AVADataset(Dataset):
    def __init__(self, csv_path, audio_dir, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.audio_dir = audio_dir
        
        self.segments = []
        self.valid_files = set()
        
        # Load CSV
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Row: video_id, start, end, label
                    if len(row) < 4: continue
                    video_id, start, end, label = row
                    
                    # Check if audio file exists
                    audio_path = os.path.join(audio_dir, f"{video_id}.wav")
                    if not os.path.exists(audio_path):
                        continue
                        
                    self.valid_files.add(audio_path)
                    
                    start_sec = float(start)
                    end_sec = float(end)
                    
                    is_speech = 1.0 if "SPEECH" in label and "NO_SPEECH" not in label else 0.0
                    
                    self.segments.append({
                        "path": audio_path,
                        "start": start_sec,
                        "end": end_sec,
                        "label": is_speech
                    })
        
        print(f"[AVADataset] Loaded {len(self.segments)} segments from {len(self.valid_files)} audio files.")
        
        # Feature Extractor
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        try:
            item = self.segments[idx]
            path = item["path"]
            start_sec = item["start"]
            end_sec = item["end"]
            is_speech = item["label"]
            
            # Load specific chunk
            # We rely on torchaudio.load frame_offset
            # Need to know original sample rate? 
            # torchaudio.info is cheap? 
            # To be safe and efficient, we can assume files are 16kHz if we preprocessed them.
            # But let's use robust loading.
            
            info = torchaudio.info(path)
            orig_sr = info.sample_rate
            
            start_frame = int(start_sec * orig_sr)
            end_frame = int(end_sec * orig_sr)
            num_frames = end_frame - start_frame
            
            if num_frames <= 0:
                # Fallback
                return torch.randn(80, 300), torch.zeros(150, 1)

            waveform, sr = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)
            
            # Resample
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or Crop to target_len
            current_len = waveform.shape[1]
            
            final_waveform = torch.zeros(1, self.target_len)
            
            valid_len = 0
            
            if current_len > self.target_len:
                # Crop (Random)
                start_crop = random.randint(0, current_len - self.target_len)
                final_waveform = waveform[:, start_crop:start_crop+self.target_len]
                valid_len = self.target_len
            else:
                # Pad (At start? Center? End?)
                # Let's put at start for simplicity
                final_waveform[:, :current_len] = waveform
                valid_len = current_len
            
            # Feature
            feature = self.mel_spectrogram(final_waveform).squeeze(0)
            if feature.shape[1] % 2 != 0:
                feature = feature[:, :-1]
            
            # Label
            # If is_speech is 1, then the valid_len part is 1.
            # Else 0.
            
            total_frames = int(self.target_len / 160) + 1 
            label_len = total_frames // 2 
            label = torch.zeros(label_len, 1)
            
            if is_speech > 0.5:
                # Calculate how many frames correspond to valid_len
                # 160 hop length * 2 (stride) = 320 effective stride for label?
                # The CRNN architecture usually reduces time dim by 2?
                # Let's assume the same logic as SyntheticVADDataset
                # effective_stride = 320
                
                effective_stride = 320
                valid_frames = int(valid_len / effective_stride)
                if valid_frames > label_len: valid_frames = label_len
                label[:valid_frames] = 1.0
            
            # Fix alignment
            curr_label_len = feature.shape[1] // 2
            if label.shape[0] != curr_label_len:
                new_label = torch.zeros(curr_label_len, 1)
                min_len = min(curr_label_len, label.shape[0])
                new_label[:min_len] = label[:min_len]
                label = new_label

            return feature, label

        except Exception as e:
            # print(f"Error loading {idx}: {e}")
            return torch.randn(80, 300), torch.zeros(150, 1)

def get_ava_dataloader(csv_path, audio_dir, batch_size=32):
    dataset = AVADataset(csv_path, audio_dir)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

def get_dataloader(speech_scp, noise_scp, batch_size=32):
    dataset = SyntheticVADDataset(speech_scp, noise_scp)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)
