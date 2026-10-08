# model/blocks/fpn_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DsBnRelu(nn.Module):
    """深度可分离卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ContextGatedConv(nn.Module):
    """上下文门控卷积（替代原来的实现）"""
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.gate_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        context = self.conv(x)
        gate = self.sigmoid(self.gate_conv(x))
        return x + context * gate


class FPN(nn.Module):
    """简化的特征金字塔网络"""
    def __init__(
        self,
        in_channels,
        out_channels=128,
        deform_groups=4,  # 参数保留但不使用
        gamma_mode="SE",
        beta_mode="contextgatedconv",
    ):
        super().__init__()
        
        # 输入通道处理
        self.in_channels = in_channels
        
        # 侧向连接
        self.lateral_convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_channel, out_channels, 1)
            )
        
        # 融合卷积
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
        
        # Gamma模块（特征增强）
        if gamma_mode == "SE":
            self.gamma_modules = nn.ModuleList([
                SEBlock(out_channels) for _ in range(len(in_channels))
            ])
        else:
            self.gamma_modules = nn.ModuleList([
                nn.Identity() for _ in range(len(in_channels))
            ])
        
        # Beta模块（特征调整）
        if beta_mode == "contextgatedconv":
            self.beta_modules = nn.ModuleList([
                ContextGatedConv(out_channels) for _ in range(len(in_channels))
            ])
        else:
            self.beta_modules = nn.ModuleList([
                nn.Identity() for _ in range(len(in_channels))
            ])
        
    def forward(self, inputs):
        """inputs: 从骨干网络输出的多尺度特征列表"""
        # 建立侧向连接
        laterals = []
        for i, (lateral_conv, x) in enumerate(zip(self.lateral_convs, inputs)):
            laterals.append(lateral_conv(x))
        
        # 自顶向下融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 上采样
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )
        
        # 应用融合卷积
        outs = []
        for i in range(used_backbone_levels):
            if i < len(self.fpn_convs):
                outs.append(self.fpn_convs[i](laterals[i]))
            else:
                outs.append(laterals[i])
        
        # 应用Gamma和Beta模块
        for i in range(len(outs)):
            outs[i] = self.gamma_modules[i](outs[i])
            outs[i] = self.beta_modules[i](outs[i])
        
        return tuple(outs)


# 简化的DINOv3Wrapper
class DINOV3Wrapper(nn.Module):
    def __init__(self, weights_path=None, device='cuda', extract_ids=[5, 11, 17, 23]):
        super().__init__()
        # 这里简化处理，实际使用时需要加载DINOv3
        print("Warning: Using simplified DINOv3Wrapper")
        self.patch_size = 16
        
    def forward(self, x):
        # 返回随机特征（仅供测试）
        B, C, H, W = x.shape
        # 模拟DINOv3的多尺度输出
        features = []
        for scale in [8, 16, 32, 64]:
            h, w = H // scale, W // scale
            feat = torch.randn(B, 1024, h, w, device=x.device) * 0.1
            features.append(feat)
        return features


# 简化的DenseAdapterLite
class DenseAdapterLite(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256, bottleneck=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, bottleneck, 1)
        self.conv2 = nn.Conv2d(bottleneck, out_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, features):
        # 假设features是列表
        adapted = []
        for feat in features:
            x = self.conv1(feat)
            x = self.relu(x)
            x = self.conv2(x)
            adapted.append(x)
        return adapted


# 简化的CBAM
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = self.sigmoid(avg_out + max_out)
        x = x * channel_attn
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.sigmoid(self.conv(spatial_attn))
        
        return x * spatial_attn