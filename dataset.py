import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
import os
import math
import time
import logging

# Configure logging to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
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
        
        # 3. Mix
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
        
        mixed = noise_chunk.clone() * 0.1
        mixed[:, start_idx:end_idx] += speech_chunk
        
        print(f"Audition Sample: Speech={os.path.basename(speech_path)}, Noise={os.path.basename(noise_path)}, SNR={snr_db:.2f}dB")
        
        return mixed, snr_db

    def __getitem__(self, idx):
        # Retry logic in case of bad files
        for retry in range(5):
            try:
                t0 = time.time()
                
                # 1. Select Noise (Background)
                noise_path = random.choice(self.noise_files)
                # Optimize: Only load the length we need
                noise_wav = self._load_audio(noise_path, target_chunk_len=self.target_len)
                if noise_wav is None: continue
                
                t1 = time.time()
                if self.verbose and retry == 0:
                    print(f"[Dataset] Loaded Noise ({t1-t0:.3f}s): {os.path.basename(noise_path)}")
                
                # Crop Noise to target length (Background base)
                noise_chunk = self._get_random_chunk(noise_wav, self.target_len)
                
                # 2. Select Speech (Foreground)
                speech_path = random.choice(self.speech_files)
                # Optimize: Only load roughly what we might need (max target_len)
                # We refine the length later
                speech_wav = self._load_audio(speech_path, target_chunk_len=self.target_len)
                if speech_wav is None: continue
                
                t2 = time.time()
                if self.verbose and retry == 0:
                    print(f"[Dataset] Loaded Speech ({t2-t1:.3f}s): {os.path.basename(speech_path)}")

                # Decide speech duration (0.5s to 2.5s)
                speech_len = speech_wav.shape[1]
                if speech_len > self.target_len:
                     # Take a sub-segment of speech
                     speech_chunk = self._get_random_chunk(speech_wav, random.randint(int(self.sample_rate*0.5), self.target_len))
                else:
                     speech_chunk = speech_wav
                
                speech_len_samples = speech_chunk.shape[1]
                
                # 3. Mix
                # Create empty canvas with noise
                # SNR: Signal to Noise Ratio. 
                # Random SNR between 5dB and 20dB
                snr_db = random.uniform(5, 20)
                
                speech_power = speech_chunk.norm(p=2)
                noise_power = noise_chunk.norm(p=2)
                
                if noise_power == 0:
                    scale = 0
                else:
                    scale = math.pow(10, -snr_db / 20) * (speech_power / noise_power)
                
                # Place speech randomly within the noise chunk
                max_start = self.target_len - speech_len_samples
                if max_start < 0: max_start = 0
                start_idx = random.randint(0, max_start)
                end_idx = start_idx + speech_len_samples
                
                # Superimpose
                # Resize noise to match the desired SNR
                mixed = noise_chunk.clone() * scale
                
                # Add speech to the noise background
                mixed[:, start_idx:end_idx] += speech_chunk
                
                if self.verbose and retry == 0:
                    print(f"[Dataset] Mixed with SNR {snr_db:.1f}dB. Total prep time: {time.time()-t0:.3f}s")

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
                label = torch.zeros(label_len, 1)
                
                # Determine active frames
                # Effective stride for the label is hop_length * 2 = 320 samples (20ms)
                effective_stride = 320
                
                start_frame = int(start_idx / effective_stride)
                end_frame = int(end_idx / effective_stride)
                
                if end_frame >= label_len: end_frame = label_len
                
                label[start_frame:end_frame] = 1.0
                
                # 5. Extract Features
                # Input: [1, Samples] -> [1, 80, Frames]
                # Remove channel dim [80, Frames]
                feature = self.mel_spectrogram(mixed).squeeze(0) 
                
                # Fix Feature/Label alignment
                # MelSpec sometimes produces +1 frame depending on padding.
                # We force it to be consistent or trim.
                if feature.shape[1] % 2 != 0:
                    feature = feature[:, :-1]
                    
                curr_label_len = feature.shape[1] // 2
                if label.shape[0] != curr_label_len:
                    # Resize label to match actual feature size
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

def get_dataloader(speech_scp, noise_scp, batch_size=32):
    dataset = SyntheticVADDataset(speech_scp, noise_scp)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)
