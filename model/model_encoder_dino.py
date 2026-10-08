import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange

# 从原模型导入必要的模块
from .blocks.fpn import FPN, DsBnRelu
from .blocks.cbam import CBAM
from .blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from .blocks.diffatts import TransformerBlock
from .blocks.refine import LearnableSoftMorph
from .backbone.efficientnetb5 import EfficientNetFeatures
from .backbone.mobilenetv2 import mobilenet_v2


class MultiTaskGatingAdapter(nn.Module):
    """
    Head-wise Multi-Task Gating Adapter
    applied INSIDE each FPN layer
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.d = dim // heads

        # 任务门控：用 GAP 做轻量任务信号
        self.det_gate = nn.Sequential(
            nn.Linear(dim, heads),
            nn.LayerNorm(heads),
            nn.Sigmoid()
        )
        self.cap_gate = nn.Sequential(
            nn.Linear(dim, heads),
            nn.LayerNorm(heads),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return:
            x_det, x_cap  (same shape)
        """
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        # global task context
        ctx = x_flat.mean(dim=1)  # (B, C)

        g_det = self.det_gate(ctx).unsqueeze(1)  # (B,1,heads)
        g_cap = self.cap_gate(ctx).unsqueeze(1)

        x_head = x_flat.view(B, -1, self.heads, self.d)

        x_det = (x_head * g_det.unsqueeze(-1)).view(B, -1, C)
        x_cap = (x_head * g_cap.unsqueeze(-1)).view(B, -1, C)

        x_det = self.drop(x_det).transpose(1, 2).view(B, C, H, W)
        x_cap = self.drop(x_cap).transpose(1, 2).view(B, C, H, W)

        return x_det, x_cap

class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128, 128],
        
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # CNN + DINO 融合
        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]),
            CBAM(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]),
            CBAM(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]),
            CBAM(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]),
            CBAM(in_dims[0], 8)
        )

        self.mtga_top = MultiTaskGatingAdapter(
                dim=in_dims[3], 
                heads=heads, 
                dropout=dropout
            )


    def forward(self, feas, ds_feas):
        """
        feas:   CNN features [x1, x2, x3, x4]
        ds_feas:DINO features [a1, a2, a3, a4]

        return:
            det_pyramid, cap_pyramid
        """
        # 1. 逐尺度融合 CNN + DINO
        x1, x2, x3, x4 = feas
        a1, a2, a3, a4 = ds_feas

        x4 = self.c4(torch.cat([x4, a4], 1))
        x3 = self.c3(torch.cat([x3, a3], 1))
        x2 = self.c2(torch.cat([x2, a2], 1))
        x1 = self.c1(torch.cat([x1, a1], 1))

        x4_det, x4_cap = self.mtga_top(x4)

        det_pyramid = [x1, x2, x3, x4_det]

        cap_feat = x4_cap

        return det_pyramid, cap_feat

class DINOv3FPNEncoder(nn.Module):
    """
    新的编码器:基于DINOv3 + CNN Backbone + FPN
    替换原有的简单CNN编码器
    """
    def __init__(
        self,
        backbone_name="mobilenetv2",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        heads=8,
        beta_mode="contextgatedconv",
        dino_weight="dinov3/pretrained/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        **kwargs
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # 1. 获取CNN骨干网络
        if backbone_name == "mobilenetv2":
            print(backbone_name)
            backbone = mobilenet_v2(pretrained=True, progress=True)
            backbone.channels = [16, 24, 32, 96, 320]
        elif backbone_name == "resnet18d":
            backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
            backbone.channels = [64, 64, 128, 256, 512]
        elif backbone_name == "efficientnet_b5":
            backbone = EfficientNetFeatures(
                bkbn_name='efficientnet_b5',
                weights='DEFAULT',
                feature_indices=[2, 3, 4, 7]
            )
            backbone.channels = [40, 64, 128, 512]  # 使用正确的通道数
        else:
            raise NotImplementedError(f"BACKBONE [{backbone_name}] is not implemented!")
        
        
        self.backbone = backbone
        
        # 2. FPN模块
        self.cnn_fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        
        # 3. DINOv3视觉编码器
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(
            weights_path=dino_weight, 
            device=device, 
            extract_ids=extract_ids
        )
        
        # 4. 稠密特征适配器
        self.dense_adp = DenseAdapterLite(
            in_dim=1024, 
            out_dim=dense_out_dim, 
            bottleneck=fpn_channels // 2
        )
        
        # 5. 特征金字塔融合
        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            heads=heads,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )
        
        # 存储输出通道数
        self.out_channels = fpn_channels

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            List[Tensor]: 4个尺度的特征图列表
        """
        # 1. CNN特征提取
        cnn_features = self.backbone.forward(x)  # 多尺度特征
        cnn_fea = self.cnn_fpn(cnn_features[-4:])  # 通过FPN得到4个尺度的特征

        # 2. DINOv3特征提取
        dino_features = self.dino(x)  # 提取DINOv3特征

        # 3. 处理稠密特征
        ds_fea = self.dense_adp(dino_features)

        # 4. 金字塔特征融合
        det_pyramid, cap_feat = self.pff(cnn_fea, ds_fea)  # 4个尺度的融合特征

        feat_list = list(det_pyramid) + [cap_feat]  # 将检测特征和描述特征组合成列表
        return feat_list

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Dynamic_conv(nn.Module):
    def __init__(self, dim):
        super(Dynamic_conv, self).__init__()
        self.d_conv_3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim
        )
        self.d_conv_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim)
        self.d_conv_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim)
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(3 * dim)
        self.conv_1 = nn.Conv2d(3 * dim, dim, 1)

    def forward(self, x):
        x1 = self.d_conv_3x3(x)
        x2 = self.d_conv_1x5(x)
        x3 = self.d_conv_5x1(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.BN(x)
        x = self.activation(x)
        x = self.conv_1(x)
        return x

class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads=8, dropout=0.):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        dim_head = attention_dim // heads
        self.scale = (attention_dim // self.heads) ** -0.5

        self.to_q = nn.Linear(dim_q, attention_dim, bias=True)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias=True)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias=True)
        self.Q_LN = nn.LayerNorm(dim_q)
        self.K_LN = nn.LayerNorm(dim_kv)
        self.V_LN = nn.LayerNorm(dim_kv)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(attention_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        #
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(1 * dim_kv, dim_kv, 1),
            nn.BatchNorm2d(dim_kv),
            nn.ReLU(),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(2 * dim_kv, dim_kv, 1),
            nn.BatchNorm2d(dim_kv),
            nn.ReLU(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x1, x2, x3):
        batch, L, c = x1.shape
        cross = not x1.equal(x2)

        h = torch.sqrt(torch.tensor(L).float()).int()
        w = torch.sqrt(torch.tensor(L).float()).int()
        x1_feat = x1.transpose(-1, 1).view(batch, c, h, w)
        x2_feat = x2.transpose(-1, 1).view(batch, c, h, w)
        x3_feat = x3.transpose(-1, 1).view(batch, c, h, w)
        x1_feat_buff = x1_feat
        if cross:
            dif = x2_feat - x1_feat
            x2_feat = self.fuse_conv(x1_feat * dif)  # + dif
            x3_feat = x2_feat  # self.fuse_conv2(x3_feat_dif)# + x3_feat

        x1 = x1_feat.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
        x2 = x2_feat.view(batch, c, -1).transpose(-1, 1)
        x3 = x3_feat.view(batch, c, -1).transpose(-1, 1)
        x1_feat_buff = x1_feat_buff.view(batch, c, -1).transpose(-1, 1)
        # add LN
        # x1 = self.Q_LN(x1)
        # x2 = self.K_LN(x2)

        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        out = out  # + x1_feat_buff
        return out  # (b,n,dim)

class Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout=0., norm_first=False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

        self.Q_d_conv = Dynamic_conv(dim_q)
        self.K_d_conv = Dynamic_conv(dim_kv)
        # self.V_d_conv = Dynamic_conv(dim_kv)

        group = dim_q
        self.PCM = nn.Sequential(
            nn.Conv2d(dim_q, dim_q, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=group),
            # the 1st convolution
            nn.BatchNorm2d(dim_q),
            nn.GELU(),
            nn.Conv2d(dim_q, dim_q, kernel_size=(1, 1), stride=1),
        )

    def forward(self, x1, x2, x3):
        batch, L, c = x1.shape
        h = torch.sqrt(torch.tensor(L).float()).int()
        w = torch.sqrt(torch.tensor(L).float()).int()
        x1_feat = x1.transpose(-1, 1).view(batch, c, h, w)
        x2_feat = x2.transpose(-1, 1).view(batch, c, h, w)
        x3_feat = x3.transpose(-1, 1).view(batch, c, h, w)
        if True:
            x1_feat = x1_feat + self.Q_d_conv(x1_feat)  # .view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
            x2_feat = x2_feat + self.K_d_conv(x2_feat)  # .view(batch, c, -1).transpose(-1, 1)
            x3_feat = x3_feat + self.K_d_conv(x3_feat)  # .view(batch, c, -1).transpose(-1, 1)
            x1 = x1_feat.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
            x2 = x2_feat.view(batch, c, -1).transpose(-1, 1)
            x3 = x3_feat.view(batch, c, -1).transpose(-1, 1)
            # res:
            res = x1_feat  # self.PCM(x1_feat)
            res = res.view(batch, c, -1).transpose(-1, 1)

        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + res  # batch, hw, c
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3) + res)  # batch, hw, c
            x = self.norm2(self.feedforward(x) + x)
        return x

class Q_Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout=0., norm_first=False):
        super(Q_Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.att2 = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim=4 * dim_q, dropout=dropout)
        self.norm0 = nn.LayerNorm(dim_q)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x1 = self.att(self.norm0(x1), self.norm0(x1), self.norm0(x1)) + x1
            x = self.att2(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x1 = self.norm0(self.att(x1, x1, x1) + x1)
            x = self.norm1(self.att(x1, x2, x3) + x1)
            x = self.norm2(self.feedforward(x) + x)
        return x

class SemanticEnhancementModule(nn.Module):
    """
    利用T1和T2特征增强消失/新增区域的语义信息
    核心思想：消失→关注T1，新增→关注T2
    """

    def __init__(self, channels):
        super().__init__()

        # 相似度计算模块
        self.similarity_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1)
        )

        # 消失区域增强：从T1提取信息
        self.disappear_enhance = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

        # 新增区域增强：从T2提取信息
        self.appear_enhance = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 1),  # 3个类别：背景、消失、新增
            nn.Softmax(dim=1)
        )

    def forward(self, T1_feat, T2_feat, dif_feature):
        """
        T1_feat: 时相1特征
        T2_feat: 时相2特征
        dif_feature: 差异特征
        返回：增强后的差异特征
        """
        B, C, H, W = T1_feat.shape

        # 1. 计算T1和T2的余弦相似度
        T1_norm = F.normalize(T1_feat, p=2, dim=1)
        T2_norm = F.normalize(T2_feat, p=2, dim=1)
        cosine_sim = (T1_norm * T2_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 2. 基于相似度生成注意力图
        # 低相似度区域可能是变化区域
        change_attention = 1 - cosine_sim

        # 3. 语义分离：消失区域关注T1，新增区域关注T2
        # 通过差异特征符号判断消失/新增
        dif_sign = torch.sign(dif_feature.mean(dim=1, keepdim=True))  # 近似判断

        # 消失区域权重（dif_sign < 0）
        disappear_weight = torch.sigmoid(-dif_sign * 5)  # 负值强化
        # 新增区域权重（dif_sign > 0）
        appear_weight = torch.sigmoid(dif_sign * 5)  # 正值强化

        # 4. 增强消失区域：从T1提取信息
        disappear_feat = self.disappear_enhance(torch.cat([T1_feat, dif_feature], dim=1))
        disappear_feat = disappear_feat * disappear_weight * change_attention

        # 5. 增强新增区域：从T2提取信息
        appear_feat = self.appear_enhance(torch.cat([T2_feat, dif_feature], dim=1))
        appear_feat = appear_feat * appear_weight * change_attention

        # 6. 门控融合
        gate_weights = self.gate(torch.cat([
            dif_feature,
            disappear_feat,
            appear_feat
        ], dim=1))  # (B, 3, H, W)

        # 加权融合
        enhanced_feat = (gate_weights[:, 0:1] * dif_feature +
                         gate_weights[:, 1:2] * disappear_feat +
                         gate_weights[:, 2:3] * appear_feat)

        return enhanced_feat, {
            'cosine_sim': cosine_sim,
            'disappear_weight': disappear_weight,
            'appear_weight': appear_weight,
            'gate_weights': gate_weights
        }

class AttentiveEncoder(nn.Module):
    """
    使用DINOv3FPNEncoder作为编码器的AttentiveEncoder
    """
    def __init__(self, n_layers, feature_size, heads, dropout=0., 
                 fpn_channels=128, decoder_dim=512, **kwargs):
        super(AttentiveEncoder, self).__init__()
        
        h_feat, w_feat, channels = feature_size
        
        # 更新channels为编码器的输出通道
        channels = fpn_channels
        feature_size = (h_feat, w_feat, channels)
        
        # change captioning branch
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        self.Dynamic_DIF_aware_TR = nn.ModuleList([])
        for i in range(n_layers):
            self.Dynamic_DIF_aware_TR.append(nn.ModuleList([
                Transformer(dim_q=channels, dim_kv=channels, heads=heads, 
                           attention_dim=channels, hidden_dim=4*channels, 
                           dropout=dropout, norm_first=False),
                Transformer(dim_q=channels, dim_kv=channels, heads=heads, 
                           attention_dim=channels, hidden_dim=4*channels, 
                           dropout=dropout, norm_first=False),
                nn.Linear(channels* 2, channels)
            ]))
            
        ## all modules related to captioning:
        self.cap_modules_list = [self.h_embedding, self.w_embedding,
                                 self.Dynamic_DIF_aware_TR]

        # change detection branch
        self.h_embedding_CD = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding_CD = nn.Embedding(w_feat, int(channels/2))
        
        # 根据编码器的输出调整dims
        # 假设编码器输出4个特征图，每个特征图的通道数相同（fpn_channels）
        dims = [channels] * 4  # 可以根据实际情况调整每个尺度的通道数
        
        self.Transformer_aug_CD = nn.ModuleList([nn.ModuleList([
            Transformer(dim_q=dims[-1], dim_kv=dims[-1], heads=heads, 
                       attention_dim=dims[-1], hidden_dim=4 * dims[-1],
                       dropout=dropout, norm_first=False),
            Transformer(dim_q=dims[-1], dim_kv=dims[-1], heads=heads, 
                       attention_dim=dims[-1], hidden_dim=4 * dims[-1],
                       dropout=dropout, norm_first=False)
        ]) for layer_num in range(3)])

        self.conv_dif = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim)
        ) for i, dim in enumerate(dims)])
        
        self.conv_fuse = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3*dim, 2*dim, 3, stride=1, padding=1), 
            nn.BatchNorm2d(2 * dim), 
            nn.ReLU(),
            nn.Conv2d(2 * dim, 2 * dim, 1)
        ) for i, dim in enumerate(dims)])

        self.cos = torch.nn.CosineSimilarity(dim=1)

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2 * dim, 2*dim, 1), 
            nn.BatchNorm2d(2*dim), 
            nn.ReLU(),
            nn.ConvTranspose2d(dim*2, 2*dims[max(i-1,0)], 4, stride=2, padding=1),
        ) for i, dim in enumerate(dims)])

        num_classes = 3  # 背景、消失、新增
        self.to_seg = nn.Sequential(
            nn.ConvTranspose2d(dims[0] * 2, dims[0], 4, stride=2, padding=1),
            nn.Conv2d(int(dims[0]), num_classes, 1),
        )

        self.semantic_enhancers = nn.ModuleList([
            SemanticEnhancementModule(dims[i]) for i in range(len(dims))
        ])
        
        self.feat_expand = nn.Sequential(
            nn.Conv2d(fpn_channels, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        # all modules related to change detection:
        self.CD_modules_list = [self.Transformer_aug_CD, self.conv_dif, 
                               self.conv_fuse, self.cos, self.to_fused, 
                               self.to_seg, self.h_embedding_CD, 
                               self.w_embedding_CD]

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos_embedding(self, x):
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        return x

    def add_pos_embedding_CD(self, x):
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding_CD(pos_h)
        embed_w = self.h_embedding_CD(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        return x
    
    def prepare_caption(self, img1, img2, CD_feat_list=None):
        batch, c, h, w = img1.shape
        img1 = img1.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
        img2 = img2.view(batch, c, -1).transpose(-1, 1)
        img_sa1, img_sa2 = img1, img2

        for (l, m, linear) in self.Dynamic_DIF_aware_TR:
            img_sa1_tr1 = l(img_sa1, img_sa2, img_sa2) #+ img_sa1
            img_sa2_tr1 = m(img_sa2, img_sa1, img_sa1) #+ img_sa2
            img_sa1 = img_sa1_tr1 #+ img_sa1
            img_sa2 = img_sa2_tr1 #+ img_sa2

        img1 = img_sa1_tr1.transpose(-1, 1).view(batch, c, h, w)
        img2 = img_sa2_tr1.transpose(-1, 1).view(batch, c, h, w)
        # prepare A B feat for detection branch
        feat_list = []
        feat_list.append(img1)
        feat_list.append(img2)
        return img1, img2, feat_list

    def change_detection(self, img1_list, img2_list, CC_feat_list = None):
        feat_num = len(img1_list)
        img_fus_list = []
        # fisrtly aug the single-temporal last features by semantic Transformer neck
        feat_1_last = img1_list[-1]
        feat_2_last = img2_list[-1]
        b, n, h, w = feat_1_last.size()

        feat_1_last = feat_1_last.view(b, n, -1).transpose(-1, 1)
        feat_2_last = feat_2_last.view(b, n, -1).transpose(-1, 1)

        # feat_1_last = self.Transformer_aug(feat_1_last)
        # feat_2_last = self.Transformer_aug(feat_2_last)#.transpose(-1, 1).view(b, n, h, w)

        feat_1_last = feat_1_last.transpose(-1, 1).view(b, n, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, n, h, w)

        img1_list[-1] = feat_1_last
        img2_list[-1] = feat_2_last

        dif_features = []
        dif_features.append(feat_1_last)
        dif_features.append(feat_2_last)
        # secondly fuse bi-temporal features in every level
        for k in range(feat_num):
            # method 1
            dif = self.conv_dif[k](img2_list[k] - img1_list[k]) + self.cos(img1_list[k], img2_list[k]).unsqueeze(1)
            if k == 3:
                dif_features.append(dif)
            fus = torch.cat([img1_list[k], dif, img2_list[k]], dim=1)
            fus = self.conv_fuse[k](fus)
            img_fus_list.append(fus)
        up = self.to_fused[-1](img_fus_list[-1])
        for i in range(feat_num-1, 0, -1): # 1,2,3
            i = i-1
            img_fus = img_fus_list[i] + up #img1_list[i] - img2_list[i]#
            img_fus = self.to_fused[i](img_fus)
            up = img_fus
            # img_fus_list.append(img_fus)
        # fused = torch.cat(img_fus_list, dim=1) # 换成Unet那种？
        seg = self.to_seg(img_fus)
        return seg, dif_features
    
    def change_detection_enhanced(self, img1_list, img2_list, CC_feat_list=None):
        feat_num = len(img1_list)
        img_fus_list = []
        semantic_maps = None

        # 1. 多尺度特征融合
        for k in range(feat_num):
            # 核心：带符号的减法
            signed_difference = img2_list[k] - img1_list[k]

            # 🔥 新增：语义增强
            if k >= 1:  # 从第2层开始使用语义增强
                enhanced_dif, semantic_maps = self.semantic_enhancers[k](
                    img1_list[k],  # T1特征
                    img2_list[k],  # T2特征
                    signed_difference  # 原始差异
                )
            else:
                enhanced_dif = signed_difference

            # 对增强后的差异进行卷积
            dif_feature = self.conv_dif[k](enhanced_dif)
            
            if k == 3:
                diff_enhance = dif_feature

            # 🔥 新增：利用语义注意力细化T1/T2特征
            if k >= 2:  # 高层特征使用更精细的语义引导
                # 计算变化注意力
                change_mask = torch.sigmoid(dif_feature.abs().mean(dim=1, keepdim=True))

                # 细化特征：变化区域增强对应时相的特征
                T1_enhanced = img1_list[k] * (1 - change_mask)  # 非变化区域用T1
                T2_enhanced = img2_list[k] * change_mask  # 变化区域用T2

                # 重新组合特征
                img1_list[k] = T1_enhanced
                img2_list[k] = T2_enhanced

            # 融合策略
            fus = torch.cat([img1_list[k], dif_feature, img2_list[k]], dim=1)
            fus = self.conv_fuse[k](fus)
            img_fus_list.append(fus)

        # 2. 解码器上采样（带语义引导）
        up = self.to_fused[-1](img_fus_list[-1])
        semantic_features = []

        for i in range(feat_num - 2, -1, -1):
            # 🔥 新增：上采样时融合语义信息
            if i < feat_num - 2:  # 不是最底层
                # 获取对应的语义增强特征
                if hasattr(self, 'semantic_memory') and i in self.semantic_memory:
                    semantic_feat = self.semantic_memory[i]
                    # 用语义特征调整上采样特征
                    semantic_attention = torch.sigmoid(semantic_feat.mean(dim=1, keepdim=True))
                    up = up * (1 + semantic_attention)  # 增强语义相关区域

            img_fus = img_fus_list[i] + up
            img_fus = self.to_fused[i](img_fus)
            up = img_fus
            semantic_features.append(img_fus)

        # 3. 输出分割图（带语义细化）
        seg = self.to_seg(up)

        return seg, diff_enhance

    def CC_neck_s0(self, feat_1_last, feat_2_last):
        feat_1_last = self.add_pos_embedding(feat_1_last)  # feat_1_last + pos_embedding
        feat_2_last = self.add_pos_embedding(feat_2_last)  # feat_2_last + pos_embedding
        b, c, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, c, -1).transpose(-1, 1)  # (b,hw,c)
        feat_2_last = feat_2_last.view(b, c, -1).transpose(-1, 1)  # (b,hw,c)

        feat_1_last = feat_1_last.transpose(-1, 1).view(b, c, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, c, h, w)

        # prepare feat for detection branch
        feat = []
        feat.append(feat_1_last)
        feat.append(feat_2_last)
        return feat_1_last, feat_2_last, feat

    def CD_neck_s0(self, img1_list, img2_list):
        feat_1_last = self.add_pos_embedding_CD(img1_list[-1])  # img1_list[-1] + pos_embedding
        feat_2_last = self.add_pos_embedding_CD(img2_list[-1])  # img2_list[-1] + pos_embedding
        b, c, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, c, -1).transpose(-1, 1)  # (b,hw,c)
        feat_2_last = feat_2_last.view(b, c, -1).transpose(-1, 1)
        for layerA, layerB in self.Transformer_aug_CD:
            feat_1_last = layerA(feat_1_last, feat_2_last, feat_2_last) + feat_1_last
            feat_2_last = layerB(feat_2_last, feat_1_last, feat_1_last) + feat_2_last
        feat_1_last = feat_1_last.transpose(-1, 1).view(b, c, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, c, h, w)
        img1_list[-1] = feat_1_last
        img2_list[-1] = feat_2_last

        # prepare CD branch feat for caption branch
        feat = []
        feat.append(img1_list[-1])
        feat.append(img2_list[-1])

        return img1_list, img2_list, feat

    def forward(self, img1_list, img2_list):
        """
        前向传播
        Args:
            imageA: 时相1图像 [B, 3, H, W]
            imageB: 时相2图像 [B, 3, H, W]
        Returns:
            img1_cap: 用于描述的特征
            img2_cap: 用于描述的特征
            seg: 变化检测分割图
            dif_features: 差异特征列表
        """
        # 1. 提取用于描述的特征（取最后一层）
        CC_img1_feat = img1_list[-1]
        CC_img2_feat = img2_list[-1]
        
        CD_img1_list = img1_list[:-1]
        CD_img2_list = img2_list[:-1]
        
        # 2. 处理检测分支
        CD_img1_list, CD_img2_list, _ = self.CD_neck_s0(CD_img1_list, CD_img2_list)
        
        # 3. 处理描述分支
        CC_img1_feat, CC_img2_feat, _ = self.CC_neck_s0(CC_img1_feat, CC_img2_feat)

        # 5. 描述分支
        img1_cap, img2_cap, CC_feat_list = self.prepare_caption(CC_img1_feat, CC_img2_feat, None)
        
        # 6. 检测分支
        seg, diff_enhance = self.change_detection_enhanced(CD_img1_list, CD_img2_list, None)

        return self.feat_expand(img1_cap), self.feat_expand(img2_cap), seg, diff_enhance

# 保持原有Encoder类的接口兼容性
class Encoder(nn.Module):
    """
    为了保持向后兼容性，包装AttentiveEncoder使其接口与原有Encoder一致
    """
    def __init__(self, network, train_stage=1, n_layers=3, 
                 feature_size=(8, 8, 512), heads=8, dropout=0.1,
                 backbone="mobilenetv2", fpn_channels=128, **kwargs):
        super().__init__()
        self.network = network
        self.attentive_encoder = AttentiveEncoder(
            train_stage=train_stage,
            n_layers=n_layers,
            feature_size=feature_size,
            heads=heads,
            dropout=dropout,
            backbone=backbone,
            fpn_channels=fpn_channels,
            **kwargs
        )
        
    def forward(self, imageA, imageB):
        """
        保持与原Encoder相同的接口
        Args:
            imageA: 时相1图像
            imageB: 时相2图像
        Returns:
            img1_list: 时相1特征列表
            img2_list: 时相2特征列表
        """
        # 通过AttentiveEncoder处理
        img1_cap, img2_cap, seg, dif_features = self.attentive_encoder(imageA, imageB)
        
        # 为了保持接口一致，返回特征列表
        # 注意：这里只返回用于描述的特征，你可能需要根据实际情况调整
        img1_list = [dif_features[0] if len(dif_features) > 0 else img1_cap]
        img2_list = [dif_features[1] if len(dif_features) > 1 else img2_cap]
        
        return img1_list, img2_list

