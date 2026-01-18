# XVAD: 基于 AVA-Speech 的声学 VAD

本项目实现了一个轻量级语音活动检测 (VAD) 模型，现在 **完全基于 AVA-Speech 数据集** 进行监督训练，不再依赖 WenetSpeech 或 MUSAN。

核心思路：利用 AVA-Speech 提供的逐秒标签（`NO_SPEECH` / `CLEAN_SPEECH` / `SPEECH_WITH_NOISE`），从对应的 YouTube 音频中裁剪片段，构造帧级别的 VAD 标签进行训练。

---

## 1. 环境依赖

建议使用 Python 3.8+，安装必要的依赖：

```bash
pip install torch torchaudio wandb numpy tqdm yt-dlp
```

`yt-dlp` 用于从 YouTube 下载 AVA-Speech 对应的视频音频。

---

## 2. 数据准备：AVA-Speech

项目目录中应该包含：

- `ava_speech_labels_v1.csv`：AVA-Speech 官方标签文件（你已经放在仓库根目录）。

### 2.1 下载音频

我们提供了脚本 [download_ava.py](file:///e:/sync/vadModel/xvad/download_ava.py) 用于根据 `ava_speech_labels_v1.csv` 自动下载对应的 YouTube 音频，并转换为 16kHz 单声道 wav：

```bash
python download_ava.py
```

脚本行为：

- 解析 `ava_speech_labels_v1.csv`，收集其中的 `video_id`。
- 对每个 `video_id`，尝试从 `https://www.youtube.com/watch?v=<video_id>` 下载音频。
- 使用 `ffmpeg`/`yt-dlp` 将其转换为：
  - 采样率：16kHz
  - 通道数：1（单声道）
- 输出文件路径：
  - `AVA_Audio/<video_id>.wav`

**注意：**

- 如果部分视频需要登录/验证（YouTube 风控），脚本会报错；你可以考虑：
  - 在本地有浏览器的环境运行此脚本，然后将 `AVA_Audio/` 目录打包上传到服务器；
  - 或者按照 `yt-dlp` 文档配置 cookies（当前脚本保持简单版本，没有自动处理 cookies）。

---

## 3. 数据集实现：AVADataset

数据加载逻辑在 [dataset.py](file:///e:/sync/vadModel/xvad/dataset.py) 中：

- `AVADataset`：直接基于 `ava_speech_labels_v1.csv` 和 `AVA_Audio/` 目录构建样本。
- 每一行 CSV 形如：

  ```text
  video_id,start_time,end_time,label
  5BDj0ow5hnA,944.79,945.4,NO_SPEECH
  ```

### 3.1 标签映射

在 `AVADataset` 中，我们将标签映射为二分类：

- `NO_SPEECH` → 0（非语音）
- `CLEAN_SPEECH` / `SPEECH_WITH_NOISE` → 1（含语音）

### 3.2 时长与对齐

- 设定固定片段长度为 `duration` 秒（默认 3s）。
- 对于每一行 `(start, end)`：
  - 使用 `torchaudio.info` 获取原始采样率；
  - 按起止时间计算 `start_frame` / `end_frame`；
  - 用 `torchaudio.load(..., frame_offset, num_frames)` 只加载该区间的音频；
  - 重采样到 16kHz、转成单声道；
  - 如果片段短于 3s，则前部填充零；长于 3s，则随机裁剪出一段 3s 的片段。

- 特征提取：
  - 使用 `MelSpectrogram`，窗口 25ms，步长 10ms，得到 80 维 mel 频谱；
  - 如果时间维度是奇数，裁掉最后一帧，以便和后续下采样对齐。

- 标签对齐：
  - 先按整段 3s 计算应该有多少帧；
  - 然后再考虑真正有效的语音长度，生成对应长度的 0/1 序列；
  - 通过对齐逻辑，保证 `feature` 时间维度与 `label` 长度匹配。

---

## 4. 训练 (Training)

训练入口为 [train_vad.py](file:///e:/sync/vadModel/xvad/train_vad.py)，现在 **只支持 AVA-Speech**：

```bash
python train_vad.py --batch_size 64 --epochs 10
```

### 4.1 单卡训练

直接运行：

```bash
python train_vad.py --batch_size 64
```

### 4.2 多卡训练 (DDP)

仍然通过 `torchrun` 启动，例如 4 卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_vad.py --batch_size 64
```

> 脚本内部会自动判断是否处于分布式环境，并使用 `DistributedDataParallel`。

### 4.3 训练细节

- **数据集**：`AVADataset`（监督式，使用 AVA-Speech 标签）。
- **模型**：`XVADModel`（在 [model.py](file:///e:/sync/vadModel/xvad/model.py) 中定义），CRNN 结构，类似 Silero VAD。
- **损失函数**：`BCELoss`。
- **日志**：使用 WandB (`project="xvad-training"`) 记录 `batch_loss` 和 `avg_loss`。
- **检查点**：每个 epoch 结束后，权重保存到 `checkpoints/xvad_epoch_*.pth`。

---

## 5. 文件说明 (Project Structure)

当前项目中与 AVA-Speech 训练直接相关的文件：

- [dataset.py](file:///e:/sync/vadModel/xvad/dataset.py)
  - `AVADataset`：AVA-Speech 监督训练数据集实现。
  - `SyntheticVADDataset`：保留的合成数据 Dataset（目前训练脚本未使用，你可以未来基于 AVA 音频自行扩展）。
- [train_vad.py](file:///e:/sync/vadModel/xvad/train_vad.py)
  - 训练主程序，默认只使用 AVA-Speech。
- [model.py](file:///e:/sync/vadModel/xvad/model.py)
  - CRNN VAD 模型结构。
- [download_ava.py](file:///e:/sync/vadModel/xvad/download_ava.py)
  - 从 YouTube 根据 `ava_speech_labels_v1.csv` 下载并预处理音频。
- [ava_speech_labels_v1.csv](file:///e:/sync/vadModel/xvad/ava_speech_labels_v1.csv)
  - AVA-Speech 官方标签文件。
- `AVA_Audio/`
  - `download_ava.py` 下载好的 wav 音频目录（默认不纳入版本控制，参见 `.gitignore`）。

---

## 6. 使用流程总结

1. 准备好 `ava_speech_labels_v1.csv`。
2. 运行：

   ```bash
   python download_ava.py
   ```

   等待音频下载并转换到 `AVA_Audio/`。

3. 启动训练：

   ```bash
   python train_vad.py --batch_size 64 --epochs 10
   ```

4. 在 WandB 中观察训练曲线，模型权重保存在 `checkpoints/` 下。

