# backbone/efficientnetb5.py
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class EfficientNetFeatures(nn.Module):
    """EfficientNet-B5 骨干网络，适配您的FPN接口"""
    
    def __init__(self, bkbn_name='efficientnet_b5', weights='DEFAULT', feature_indices=[2, 3, 4, 7]):
        """
        Args:
            output_layer: 输出到第几层（0-8）
                        0: stem
                        1-7: blocks[0-6]
                        8: 最后一个conv
        """
        super().__init__()
        
        # 加载完整模型
        entire_model = getattr(torchvision.models, bkbn_name)(weights=weights)
        
        # 获取features部分
        self.features = entire_model.features
        
        self.feature_indices = feature_indices
        
    def forward(self, x):
        """
        返回多尺度特征
        注意：我们需要从第1层开始（跳过stem），然后取最后4层
        """
        features = []
        
        # 手动遍历features的每个模块
        for i, module in enumerate(self.features):
            x = module(x)

            if i in self.feature_indices:
                features.append(x)
            
            if len(features) == len(self.feature_indices):
                break
        
        assert len(features) == 4, f"Expected 4 features, got {len(features)}"
        
        return features