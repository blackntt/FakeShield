import torch
import torch.nn as nn
from .DnCNN import DnCNN

class NoiseprintProjector(nn.Module):
    def __init__(self, weights_path="weights/np_plus_plus.pt", mm_hidden_size=4096, num_tokens=64):
        super().__init__()
        # 1. Load Noiseprint++ Pre-trained (Frozen)
        self.noiseprint = self._load_noiseprint(weights_path)
        
        # 2. Feature Extractor nhẹ: Biến Noise Map 1-channel -> 512 channels
        self.conv_adapter = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),   # [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1), # [B, 256, H/4, W/4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),# [B, 512, H/8, W/8]
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # 3. Pooling & Projection: Nén không gian về 64 tokens và chiếu lên dim của LLM
        self.pool = nn.AdaptiveAvgPool2d((8, 8)) # 8x8 = 64 tokens
        self.proj = nn.Sequential(
            nn.Linear(512, mm_hidden_size),
            nn.GELU(),
            nn.Linear(mm_hidden_size, mm_hidden_size)
        )

    def _load_noiseprint(self, weights_path):
        model = DnCNN(
            nplanes_in=1,
            nplanes_out=1,
            features=64,
            kernel=3,
            depth=17,
            activation='relu',
            residual=False,  # Đổi sang False nếu checkpoint gốc không dùng residual
            bn=False         # Đổi sang False nếu checkpoint gốc không dùng BatchNorm
        )

        state_dict = torch.load(weights_path, map_location='cpu')

        # 1. Bóc tách các wrapper lồng nhau trong checkpoint
        if 'noiseprint' in state_dict:
            state_dict = state_dict['noiseprint']
        if 'network' in state_dict:
            state_dict = state_dict['network']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # 2. Xóa các tiền tố thừa (ví dụ 'module.' hoặc 'model.') nếu có
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('model.', '')
            new_state_dict[name] = v

        # 3. Load trọng số với strict=False để bỏ qua các key thừa như global_step
        model.load_state_dict(new_state_dict, strict=False)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, images):
        # Chuyển RGB sang Grayscale nếu ảnh đầu vào là 3 channels
        if images.shape[1] == 3:
            # Công thức chuẩn RGB to Gray
            gray_images = 0.2989 * images[:, 0:1, :, :] + 0.5870 * images[:, 1:2, :, :] + 0.1140 * images[:, 2:3, :, :]
        else:
            gray_images = images

        # 1. Trích xuất Noiseprint (Không tính gradient cho Noiseprint)
        with torch.no_grad():
            noise_map = self.noiseprint(gray_images) # [B, 1, H, W]

        # 2. Extract features & Project (Có tính gradient để train Adapter)
        feats = self.conv_adapter(noise_map)                   # [B, 512, H/8, W/8]
        pooled = self.pool(feats).flatten(2).permute(0, 2, 1)  # [B, 64, 512]
        freq_tokens = self.proj(pooled)                        # [B, 64, 4096]

        return freq_tokens