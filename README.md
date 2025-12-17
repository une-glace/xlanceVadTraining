# XVAD: 基于合成训练的轻量级声学 VAD

本项目实现了一个基于 **合成数据生成 (Synthetic Data Generation)** 训练的轻量级语音活动检测 (VAD) 模型。它在训练过程中动态混合纯净语音（如 WenetSpeech）和背景噪声（如 MUSAN），无需人工标注的时间戳。

## 1. 环境依赖

安装必要的 Python 库：

```bash
pip install torch torchaudio wandb numpy
```

## 2. 数据准备

你需要准备两个源数据集：
1.  **语音数据集 (Speech Dataset)**：纯净语音（例如 WenetSpeech, LibriSpeech）。
2.  **噪声数据集 (Noise Dataset)**：背景噪声（例如 MUSAN, AudioSet）。

由于这些数据集通常很大，我们首先扫描它们以生成文件列表（`.scp` 文件）。

运行 `prepare_list.py` 脚本：

```bash
# 示例命令（请修改路径以匹配你的服务器）
python prepare_list.py \
  --speech_dir /path/to/WenetSpeech/data/audio \
  --noise_dir /path/to/musan/noise \
  --output_dir .
```

这将在当前目录下生成两个文件：
*   `speech.scp`: 语音音频文件列表。
*   `noise.scp`: 噪声音频文件列表。

## 3. 训练

训练脚本 `train_vad.py` 会自动加载 `.scp` 文件，执行在线混合，并将指标记录到 [WandB](https://wandb.ai/)。

### 开始训练

#### 单卡训练 (Single GPU)
```bash
python train_vad.py
```

#### 多卡训练 (Multi-GPU)
推荐使用 `torchrun` 启动：

```bash
# --nproc_per_node: 使用的 GPU 数量
# CUDA_VISIBLE_DEVICES: 指定具体的 GPU 编号
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_vad.py --batch_size 128
```

*注意：首次运行时，WandB 会要求输入你的 API Key。*

### 训练细节
*   **数据集**: `SyntheticVADDataset` (在 `dataset.py` 中) 动态混合语音和噪声，随机信噪比 (SNR) 为 5dB - 20dB。
*   **模型**: `XVADModel` (在 `model.py` 中) 是一个受 Silero VAD 启发的 CRNN (Conv1d + LSTM) 模型。
*   **日志**: Loss 和指标将记录到 WandB 项目 `xvad-training` 中。
*   **检查点**: 每个 epoch 结束后模型会保存到 `checkpoints/` 目录。

## 项目结构

*   `dataset.py`: 定义用于在线数据合成的 `SyntheticVADDataset` 类。
*   `model.py`: CRNN 模型架构。
*   `prepare_list.py`: 用于扫描目录并创建文件列表的辅助脚本。
*   `train_vad.py`: 集成了 WandB 的主训练循环。

---

# XVAD: Lightweight Acoustic VAD with Synthetic Training

This project implements a lightweight Voice Activity Detection (VAD) model trained using **Synthetic Data Generation**. It dynamically mixes clean speech (e.g., WenetSpeech) with background noise (e.g., MUSAN) during training, eliminating the need for manually labeled timestamps.

## 1. Requirements

Install the necessary Python libraries:

```bash
pip install torch torchaudio wandb numpy
```

## 2. Data Preparation

You need two source datasets:
1.  **Speech Dataset**: Clean speech (e.g., WenetSpeech, LibriSpeech).
2.  **Noise Dataset**: Background noise (e.g., MUSAN, AudioSet).

Since these datasets are usually large, we first scan them to generate file lists (`.scp` files).

Run the `prepare_list.py` script:

```bash
# Example command (modify paths to match your server)
python prepare_list.py \
  --speech_dir /path/to/WenetSpeech/data/audio \
  --noise_dir /path/to/musan/noise \
  --output_dir .
```

This will generate two files in the current directory:
*   `speech.scp`: List of speech audio files.
*   `noise.scp`: List of noise audio files.

## 3. Training

The training script `train_vad.py` automatically loads the `.scp` files, performs on-the-fly mixing, and logs metrics to [WandB](https://wandb.ai/).

### Start Training
#### Single GPU
```bash
python train_vad.py
```

#### Multi-GPU
It is recommended to use `torchrun`:

```bash
# --nproc_per_node: Number of GPUs to use
# CUDA_VISIBLE_DEVICES: Specify GPU IDs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_vad.py --batch_size 128
```

*Note: On the first run, WandB will ask for your API key.*

### Training Details
*   **Dataset**: `SyntheticVADDataset` (in `dataset.py`) dynamically mixes speech and noise with random SNRs (5dB - 20dB).
*   **Model**: `XVADModel` (in `model.py`) is a CRNN (Conv1d + LSTM) inspired by Silero VAD.
*   **Logging**: Loss and metrics are logged to WandB project `xvad-training`.
*   **Checkpoints**: Models are saved to `checkpoints/` after every epoch.

## Project Structure

*   `dataset.py`: Defines the `SyntheticVADDataset` class for on-the-fly data synthesis.
*   `model.py`: The CRNN model architecture.
*   `prepare_list.py`: Helper script to scan directories and create file lists.
*   `train_vad.py`: Main training loop with WandB integration.
