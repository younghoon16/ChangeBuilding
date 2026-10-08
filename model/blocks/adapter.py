import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os

REPO_DIR = "dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12,
    "VITSP": 12,
    "VITB": 12,
    "VITL": 24,
    "VITHP": 32,
    "VIT7B": 40,
}


class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        weights_path="dinov3/pretrained/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
    ):
        super().__init__()
        self.device = device
        # 1. 首先创建不加载预训练权重的模型
        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            pretrained=False,  # 不自动下载
        )
        # 2. 手动加载权重文件
        if os.path.exists(weights_path):
            print(f"从本地加载DINOv3权重: {weights_path}")
            
            # 加载权重
            state_dict = torch.load(weights_path, map_location="cpu")
            
            # 检查权重格式
            is_hf_format = any("layer." in key or "embeddings." in key for key in state_dict.keys())
            
            if is_hf_format:
                print("检测到HuggingFace格式权重，进行转换...")
                # 转换HF格式为torch.hub格式
                converted_state_dict = self._convert_hf_to_torchhub(state_dict)
                
                # 非严格模式加载
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    converted_state_dict, strict=False
                )
                
                if missing_keys:
                    print(f"警告: 缺失 {len(missing_keys)} 个键")
                if unexpected_keys:
                    print(f"警告: 意外 {len(unexpected_keys)} 个键")
            else:
                # 直接加载
                self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"警告: 权重文件不存在 {weights_path}")
            print("将使用随机初始化的DINOv3模型")
        self.model = self.model.eval().to(device)
        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = extract_ids

        # freeze the backbone
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        scale_factor = 2 / (512 / x.shape[-1])
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    x, n=range(self.n_layers), reshape=True, norm=True
                )
                feats_ = []
                for i in range(len(self.extract_ids)):
                    feats_.append(
                        F.interpolate(
                            feats[self.extract_ids[i]],
                            scale_factor=scale_factor,
                            mode="bilinear",
                        )
                    )
        return feats_


class SepAdapterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 64, act=nn.SiLU):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                r, r, kernel_size=3, padding=1, groups=r, bias=False
            ),  # depthwise
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return x


class DenseAdapterLite(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        bottleneck=64,
        share=False,
    ):
        super().__init__()
        if share:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck) for _ in range(4)]
            )
        self.share = share

    def forward(self, feats):
        """
        feats: list of 4 tensors, each [B, C, H_i, W_i]（C = in_dim）
        return: list of 4 tensors, each [B, out_dim, S_i, S_i], S_i ∈ self.sizes
        """
        outs = []
        for i, x in enumerate(feats):
            x = F.interpolate(
                x,
                scale_factor=2 / (2**i),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            outs.append(block(x))
        return outs
