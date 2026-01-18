# XVAD: 基于 Kaggle + AVA-Speech 的声学 VAD

本项目实现了一个轻量级语音活动检测 (VAD) 模型，现在 **以 Kaggle 竞赛数据集为主、AVA-Speech 为辅** 进行监督训练：

- 主数据集：Kaggle 比赛 **Voice Activity Detection (SJTU Spring 2023)**  
  (`voice-activity-detection-sjtu-spring-2023`)
- 补充数据集：AVA-Speech（从 YouTube 下载音频）

核心思路：利用两套数据集提供的语音活动标注（逐帧/逐时间段的 speech / non-speech），在统一的特征与标签对齐方式下训练一个 CRNN VAD 模型。

---

## 1. 环境依赖

建议使用 Python 3.8+，安装必要的依赖：

```bash
pip install torch torchaudio wandb numpy tqdm yt-dlp
```

`yt-dlp` 用于从 YouTube 下载 AVA-Speech 对应的视频音频。

---

## 2. 数据准备

### 2.1 Kaggle VAD 数据集（主数据）

项目假设你已经从 Kaggle 下载并解压了比赛数据，目录结构类似：

- `voice-activity-detection-sjtu-spring-2023/`
  - `vad/`
    - `data/`
      - `train_label.txt`
      - `dev_label.txt`
    - `wavs/`
      - `100-122655-0035.wav`
      - `1001-134707-0047.wav`
      - `...`

其中 `train_label.txt` 的每一行形如：

```text
100-122655-0035 0.14,1.79 1.82,2.88 3.3,3.97 3.98,4.42 4.43,6.86 7.79,11.0 11.43,13.72
```

含义：

- `100-122655-0035`：对应音频文件 `wavs/100-122655-0035.wav`
- 后面是一串 `start,end` 的时间对，表示在这些区间内为「有语音」，其余时间为「无语音」

**默认配置：**

- 训练脚本默认使用：
  - 标签文件：`voice-activity-detection-sjtu-spring-2023/vad/data/train_label.txt`
  - 音频目录：`voice-activity-detection-sjtu-spring-2023/vad/wavs`

如果你放在其他位置，可以在训练时通过参数 `--kaggle_label` 和 `--kaggle_audio_dir` 修改。

### 2.2 AVA-Speech（可选补充）

项目目录中应该包含：

- `ava_speech_labels_v1.csv`：AVA-Speech 官方标签文件（你已经放在仓库根目录）。

#### 2.2.1 下载 AVA 音频

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

## 3. 数据集实现：Dataset 设计

数据加载逻辑在 [dataset.py](file:///e:/sync/vadModel/xvad/dataset.py) 中，主要包含三个 Dataset：

- `KaggleVADDataset`：基于 Kaggle 比赛数据的主监督训练集
- `AVADataset`：基于 AVA-Speech 的补充训练集
- `SyntheticVADDataset`：基于语音+噪声混合的合成数据（当前训练脚本默认不使用）

### 3.1 KaggleVADDataset（主数据集）

- 入口参数：
  - `label_path`：如 `voice-activity-detection-sjtu-spring-2023/vad/data/train_label.txt`
  - `audio_dir`：如 `voice-activity-detection-sjtu-spring-2023/vad/wavs`
  - `sample_rate=16000`，`duration=3.0`
- 初始化时：
  - 逐行读取 `train_label.txt`，解析出：
    - `utt_id`（例如 `100-122655-0035`）
    - 时间段字符串（例如 `"0.14,1.79 1.82,2.88 ..."`）
  - 根据 `utt_id` 在 `audio_dir` 下寻找 `utt_id.wav` 或 `utt_id.flac`
  - 使用内部的 `parse_kaggle_vad_label` 将时间段转成帧级 0/1 序列
- `__getitem__`：
  - 从整段音频中随机截取一个 3 秒片段，不足则在尾部补零
  - 提取 80 维 mel 频谱特征（25ms 窗，10ms 步长）
  - 按帧对齐标签，并按照模型时间下采样（时间维 /2）聚合成帧级 0/1 标签

这个 Dataset 是当前训练流程的默认数据源。

### 3.2 AVADataset（AVA-Speech 补充数据）

- 直接基于 `ava_speech_labels_v1.csv` 和 `AVA_Audio/`。
- 每一行 CSV 形如：

  ```text
  video_id,start_time,end_time,label
  5BDj0ow5hnA,944.79,945.4,NO_SPEECH
  ```

- 标签映射：
  - `NO_SPEECH` → 0（非语音）
  - `CLEAN_SPEECH` / `SPEECH_WITH_NOISE` → 1（含语音）
- `__getitem__`：
  - 按 `(start, end)` 从对应视频音频中裁剪出一个片段
  - 重采样到 16k 单声道、pad/crop 到固定 3 秒
  - 提 mel 特征，并按照与 Kaggle 一致的方式对齐生成帧级标签

在训练脚本中，可以将其作为补充数据与 Kaggle 合并（`--dataset kaggle_ava`）。

### 3.3 SyntheticVADDataset（合成数据）

- 通过两份列表：
  - `speech_scp`：语音文件列表（如 WenetSpeech 切分后）
  - `noise_scp`：噪声文件列表（如 MUSAN）
- 在线随机生成：
  - 纯噪声样本
  - 纯语音样本
  - 语音+噪声混合样本（随机 SNR 5–20 dB）
- 并按照与模型一致的时间分辨率生成帧级标签。

当前训练脚本默认不使用该 Dataset，仅在 [audition.py](file:///e:/sync/vadModel/xvad/audition.py) 中用于试听和调试。

---

## 4. 训练 (Training)

训练入口为 [train_vad.py](file:///e:/sync/vadModel/xvad/train_vad.py)。  
现在支持三种模式：

- 只用 Kaggle：`--dataset kaggle`（默认）
- 只用 AVA：`--dataset ava`
- Kaggle + AVA 混合：`--dataset kaggle_ava`

### 4.1 只用 Kaggle 训练（默认）

在项目根目录直接运行：

```bash
python train_vad.py --batch_size 64 --epochs 10
```

等价于：

```bash
python train_vad.py \
  --dataset kaggle \
  --kaggle_label voice-activity-detection-sjtu-spring-2023/vad/data/train_label.txt \
  --kaggle_audio_dir voice-activity-detection-sjtu-spring-2023/vad/wavs \
  --batch_size 64 --epochs 10
```

### 4.2 Kaggle + AVA 混合训练

在已准备好 AVA 音频的前提下（见 2.2），可以使用：

```bash
python train_vad.py --dataset kaggle_ava --batch_size 64 --epochs 10
```

此时训练集为：

- KaggleVADDataset(label_path, audio_dir)
- AVADataset(ava_speech_labels_v1.csv, AVA_Audio)

通过 `ConcatDataset` 合并成一个大的训练集。

### 4.3 只用 AVA 训练（兼容旧流程）

如果你想仅基于 AVA-Speech 训练，可以使用：

```bash
python train_vad.py --dataset ava --batch_size 64 --epochs 10
```

要求：

- `ava_speech_labels_v1.csv` 在项目根目录
- `AVA_Audio/` 中存在对应的 `<video_id>.wav` 文件（通过 `download_ava.py` 下载）

### 4.4 单卡 / 多卡训练

- 单卡：

  ```bash
  python train_vad.py --batch_size 64
  ```

- 多卡 (DDP)，例如 4 卡：

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_vad.py --batch_size 64
  ```

脚本会自动判断是否处于分布式环境，并使用 `DistributedDataParallel`。

### 4.5 训练细节

- **数据集**：
  - 默认：`KaggleVADDataset`
  - 可选补充：`AVADataset`，通过 `ConcatDataset` 混合
- **模型**：`XVADModel`（在 [model.py](file:///e:/sync/vadModel/xvad/model.py) 中定义），CRNN 结构，类似 Silero VAD。
- **损失函数**：`BCELoss`。
- **日志**：使用 WandB (`project="xvad-training"`) 记录 `batch_loss` 和 `avg_loss`。
- **检查点**：每个 epoch 结束后，权重保存到 `checkpoints/xvad_epoch_*.pth`。

---

## 5. 文件说明 (Project Structure)

与 VAD 训练直接相关的文件：

- [dataset.py](file:///e:/sync/vadModel/xvad/dataset.py)
  - `KaggleVADDataset`：基于 Kaggle 比赛数据的主训练 Dataset。
  - `AVADataset`：AVA-Speech 监督训练 Dataset，实现与 Kaggle 一致的标签对齐逻辑。
  - `SyntheticVADDataset`：保留的合成数据 Dataset（当前训练脚本默认不使用）。
- [train_vad.py](file:///e:/sync/vadModel/xvad/train_vad.py)
  - 训练主程序，支持 `kaggle` / `ava` / `kaggle_ava` 三种模式。
- [model.py](file:///e:/sync/vadModel/xvad/model.py)
  - CRNN VAD 模型结构。
- [download_ava.py](file:///e:/sync/vadModel/xvad/download_ava.py)
  - 从 YouTube 根据 `ava_speech_labels_v1.csv` 下载并预处理音频。
- [ava_speech_labels_v1.csv](file:///e:/sync/vadModel/xvad/ava_speech_labels_v1.csv)
  - AVA-Speech 官方标签文件。
- `voice-activity-detection-sjtu-spring-2023/vad/data/train_label.txt`
  - Kaggle 比赛提供的训练集 VAD 时间段标签。
- `voice-activity-detection-sjtu-spring-2023/vad/wavs/`
  - Kaggle 数据集对应的 wav 音频目录。
- `AVA_Audio/`
  - `download_ava.py` 下载好的 wav 音频目录（默认不纳入版本控制，参见 `.gitignore`）。

---

## 6. 使用流程示例

### 6.1 仅使用 Kaggle 数据集训练

1. 从 Kaggle 下载并解压 `voice-activity-detection-sjtu-spring-2023` 到项目根目录。
2. 确认存在：
   - `voice-activity-detection-sjtu-spring-2023/vad/data/train_label.txt`
   - `voice-activity-detection-sjtu-spring-2023/vad/wavs/*.wav`
3. 启动训练：

   ```bash
   python train_vad.py --batch_size 64 --epochs 10
   ```

4. 在 WandB 中观察训练曲线，模型权重保存在 `checkpoints/` 下。

### 6.2 使用 Kaggle + AVA-Speech 混合训练

1. 完成 Kaggle 数据准备（同上）。
2. 准备 AVA-Speech：
   - 将 `ava_speech_labels_v1.csv` 放到项目根目录。
   - 运行 `python download_ava.py` 下载 `AVA_Audio/*.wav`。
3. 启动混合训练：

   ```bash
   python train_vad.py --dataset kaggle_ava --batch_size 64 --epochs 10
   ```

4. 查看 WandB 日志和 `checkpoints/` 中的模型权重。

