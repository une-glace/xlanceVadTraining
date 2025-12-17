import torch
import torch.nn as nn

class XVADModel(nn.Module):
    def __init__(self, input_dim=80, hidden_size=64):
        """
        A lightweight CRNN VAD model inspired by TEN VAD.
        受到 TEN VAD 启发的轻量级 CRNN VAD 模型。
        
        Args/参数:
            input_dim (int): Input feature dimension (e.g., 80 for Log-Mel Spectrogram).
                             输入特征维度（例如 Log-Mel 频谱图通常为 80）。
            hidden_size (int): Hidden size for LSTM layers (e.g., 64).
                               LSTM 层的隐藏层大小（例如 64）。
        """
        super(XVADModel, self).__init__()
        
        # 1. Feature Extractor (CNN) / 特征提取器 (CNN)
        # Input: [Batch, Input_Dim, Time] -> Output: [Batch, Channels, Time]
        # 输入: [批次大小, 特征维度, 时间步] -> 输出: [批次大小, 通道数, 时间步]
        # We use 1D Conv to extract local spectral patterns across frequency bands.
        # 我们使用一维卷积在频率维度上提取局部声学模式（纹理）。
        self.conv = nn.Sequential(
            # Layer 1
            # In: [B, 80, T] -> Out: [B, 32, T]
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # In: [B, 32, T] -> Out: [B, 32, T/2]
            nn.MaxPool1d(kernel_size=2, stride=2), # Downsample time by 2 / 时间维度下采样 2 倍
            
            # Layer 2
            # In: [B, 32, T/2] -> Out: [B, 64, T/2]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 2. Sequential Memory (LSTM) / 时序记忆 (LSTM)
        # Input shape needs to be permuted: [Batch, Time, Features]
        # 输入形状需要转换维度: [批次大小, 时间步, 特征数]
        self.lstm = nn.LSTM(
            input_size=64,       # Matches output channels of CNN / 对应 CNN 输出的通道数
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        
        # 3. Classifier (Head) / 分类头
        # Input: [Batch, Time, Hidden_Size] -> Output: [Batch, Time, 1]
        # 输入: [批次大小, 时间步, 隐藏层大小] -> 输出: [批次大小, 时间步, 1]
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output probability 0~1 / 输出 0~1 的概率
        )
        
    def forward(self, x, hidden=None):
        """
        Forward pass for streaming or batch inference.
        流式或批量推理的前向传播。
        
        Args/参数:
            x (torch.Tensor): Input features [Batch, Input_Dim, Time]
                              输入特征 [批次大小, 输入维度(80), 原始时间步]
            hidden (tuple): (h_0, c_0) for LSTM state. None for initial state.
                            LSTM 的状态 (h_0, c_0)。如果是初始状态则为 None。
            
        Returns/返回:
            out (torch.Tensor): Probability [Batch, Time/2, 1]
                                概率 [批次大小, 下采样后的时间步, 1]
            hidden (tuple): Updated (h_n, c_n) / 更新后的状态
        """
        # ==========================================
        # Step 1: CNN Feature Extraction / CNN 特征提取
        # ==========================================
        # Input / 输入: [Batch, 80, T]
        x = self.conv(x) 
        # Output / 输出: [Batch, 64, T/2] 
        # (Channels became 64, Time became T/2 due to MaxPool / 通道变64，时间因池化变一半)
        
        # ==========================================
        # Step 2: Dimension Permutation / 维度转换
        # ==========================================
        # LSTM requires [Batch, Time, Features] / LSTM 需要 [批次, 时间, 特征]
        x = x.permute(0, 2, 1) 
        # Output / 输出: [Batch, T/2, 64]
        
        # ==========================================
        # Step 3: LSTM Memory Processing / LSTM 记忆处理
        # ==========================================
        # Input / 输入: [Batch, T/2, 64]
        x, hidden = self.lstm(x, hidden)
        # Output / 输出: [Batch, T/2, Hidden_Size(64)]
        
        # ==========================================
        # Step 4: Classification / 分类
        # ==========================================
        # Input / 输入: [Batch, T/2, 64]
        out = self.fc(x)
        # Output / 输出: [Batch, T/2, 1]
        
        return out, hidden

if __name__ == "__main__":
    # Test the model with dummy data
    model = XVADModel()
    dummy_input = torch.randn(1, 80, 100) # [Batch, Freq, Time]
    output, _ = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [1, 50, 1] (Time is halved)
