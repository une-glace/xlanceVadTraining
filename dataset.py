from typing import Any
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
import os
import json
import math
import time
import logging
import csv
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class SyntheticVADDataset(Dataset):
    def __init__(self, speech_scp, noise_scp, label_path='wenet_special.json',sample_rate=16000, duration=3.0, epoch_len=10000, verbose=False):
        """
        Args:
            speech_scp (str): Path to speech file list.
            noise_scp (str): Path to noise file list.
            label_path(str): Path to label.
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
        with open(label_path,'r',encoding='utf-8') as f:
            self.label = json.load(f)
        
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
    
    def convert_label(self,audio_id,start_time,len):
        segments = self.label[audio_id]["segments"]
        start = torch.tensor([segment["begin_time"]] for segment in segments)
        end = torch.tensor([segment["end_time"]] for segment in segments)
        assert len(start)==len(end)
        durations = start-end
        label = torch.zeros(self.label[audio_id]["duration"]*self.sample_rate)
        for i,duration in zip(start,durations):
            label[i*self.sample_rate:(i+duration)*self.sample_rate] = 1
        return label[start_time:start_time+len]

    def _get_random_chunk(self, waveform, target_len):
        """Extract a random chunk of target_len from waveform."""
        if waveform.shape[1] < target_len:
            # Pad if too short
            padding = target_len - waveform.shape[1]
            satrt = 0
            return torch.nn.functional.pad(waveform, (0, padding)),start
        else:
            # Random crop
            start = random.randint(0, waveform.shape[1] - target_len)
            return waveform[:, start:start+target_len],start

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
                t1 = time.time()
                
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
                        noise_chunk,_ = self._get_random_chunk(noise_wav, self.target_len)
                        
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
                         speech_chunk,start = self._get_random_chunk(speech_wav, random.randint(int(self.sample_rate*0.5), self.target_len))
                    else:
                         speech_chunk,start = speech_wav,0
                    
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
                         speech_chunk,start = self._get_random_chunk(speech_wav, random.randint(int(self.sample_rate*0.5), self.target_len))
                    else:
                         speech_chunk,start = speech_wav,0
                    
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
                # Resolution: The model output time dimension is reduced by 2 (MaxPool stride 2).
                # MelSpectrogram hop_length=160 (10ms).
                # Total frames = target_len / 160.
                # Model output frames = Total frames / 2.
                
                total_frames = int(self.target_len / 160) + 1 # +1 usually due to center=True in STFT
                # Correction: let's verify feature shape later. 
                # For 3s (48000 samples), hop 160 -> 300 frames.
                # Model output -> 150 frames.
                
                # Construct frame-level label
                # We need to map sample indices to frame indices.
                # Frame index i covers samples around i * hop_length.
                
                label_len = total_frames // 2 # Because of MaxPool in model
                label = torch.zeros(label_len,1)
                # Determine active frames
                # Effective stride for the label is hop_length * 2 = 320 samples (20ms)
                effective_stride = 320
                
                start_frame = int(start_idx / effective_stride)
                end_frame = int(end_idx / effective_stride)

                if end_frame >= label_len: end_frame = label_len

                speech_label =self.convert_label(speech_path.split('/'[-1].split('.')[0]),start,speech_len_samples)
                for i in range(0,speech_len_samples,effective_stride): #320->MEl-hop_len, 400->MEl-win_len
                    ratio = float(torch.mean(speech_label[i:i+400]))
                    if ratio > 0.5: label[start_frame+1,:] = 1.0 #threshold=0.5,can tuning
                
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

class KaggleVADDataset(Dataset):
    def __init__(self, label_path, audio_dir, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.audio_dir = audio_dir
        self.frame_shift = 0.01
        self.items = []
        path_index = {}

        if os.path.exists(self.audio_dir):
            for root, _, files in os.walk(self.audio_dir):
                for name in files:
                    lower = name.lower()
                    if lower.endswith(".wav") or lower.endswith(".flac"):
                        key, _ = os.path.splitext(name)
                        full_path = os.path.join(root, name)
                        path_index[key] = full_path
        num_audio_files = len(path_index)
        spans_per_utt = defaultdict(list)

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) < 2:
                        continue
                    utt_id, segs = parts
                    for pair in segs.split():
                        try:
                            start, end = map(float, pair.split(","))
                        except Exception:
                            continue
                        if end <= start:
                            continue
                        spans_per_utt[utt_id].append((start, end))
        num_utts = len(spans_per_utt)
        for utt_id, spans in spans_per_utt.items():
            audio_path = path_index.get(utt_id)
            if audio_path is None:
                continue
            try:
                info = torchaudio.info(audio_path)
                orig_sr = info.sample_rate
                total_sec = info.num_frames / float(orig_sr)
                total_frames = int(total_sec / self.frame_shift) + 1
                labels = [0] * total_frames
                for start_sec, end_sec in spans:
                    s = max(0.0, start_sec)
                    e = min(total_sec, end_sec)
                    if e <= s:
                        continue
                    start_frame = int(s / self.frame_shift)
                    end_frame = int(e / self.frame_shift)
                    if end_frame >= total_frames:
                        end_frame = total_frames - 1
                    for i in range(start_frame, end_frame + 1):
                        labels[i] = 1
                self.items.append(
                    {
                        "utt": utt_id,
                        "path": audio_path,
                        "labels": labels,
                    }
                )
            except Exception:
                continue
        logger.info(
            f"[KaggleVADDataset] Found {num_audio_files} audio files, {num_utts} utts, loaded {len(self.items)} items from audio_dir={self.audio_dir}, label_path={label_path}"
        )
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )

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
    def __init__(self, lab_dir, audio_dir, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.audio_dir = audio_dir
        self.frame_shift = 0.01
        self.items = []
        spans_per_audio = defaultdict(list)

        if os.path.isdir(lab_dir):
            for name in os.listdir(lab_dir):
                if not name.endswith(".lab"):
                    continue
                lab_path = os.path.join(lab_dir, name)
                base = os.path.splitext(name)[0]
                if "_c_" in base:
                    video_id = base.split("_c_")[0]
                else:
                    video_id = base
                audio_path = os.path.join(audio_dir, f"{video_id}.wav")
                if not os.path.exists(audio_path):
                    continue
                with open(lab_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        start = parts[0]
                        end = parts[1]
                        tag = parts[2]
                        if tag.lower() != "speech":
                            continue
                        start_sec = float(start)
                        end_sec = float(end)
                        spans_per_audio[audio_path].append((start_sec, end_sec))
        for audio_path, spans in spans_per_audio.items():
            try:
                info = torchaudio.info(audio_path)
                orig_sr = info.sample_rate
                total_sec = info.num_frames / float(orig_sr)
                total_frames = int(total_sec / self.frame_shift) + 1
                labels = [0] * total_frames
                for start_sec, end_sec in spans:
                    s = max(0.0, start_sec)
                    e = min(total_sec, end_sec)
                    if e <= s:
                        continue
                    start_frame = int(s / self.frame_shift)
                    end_frame = int(e / self.frame_shift)
                    if end_frame >= total_frames:
                        end_frame = total_frames - 1
                    for i in range(start_frame, end_frame + 1):
                        labels[i] = 1
                self.items.append({"path": audio_path, "labels": labels})
            except Exception:
                continue
        logger.info(f"[AVADataset] Loaded {len(self.items)} audio files from {audio_dir}")
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )

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
            
def get_ava_dataloader(lab_dir, audio_dir, batch_size=32):
    if not os.path.isdir(audio_dir):
        raise ValueError(f"Audio directory {audio_dir} does not exist.")
    dataset = AVADataset(lab_dir, audio_dir)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

def get_dataloader(speech_scp, noise_scp, batch_size=32):
    dataset = SyntheticVADDataset(speech_scp, noise_scp)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)
