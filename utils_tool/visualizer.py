import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.rcParams['figure.max_open_warning'] = 10
import torch
import numpy as np
import os


def visualize_dif_features(dif_features, pred, gt, name, args):
    """
    可视化差异特征：第一行显示Image A, Image B, Prediction
                第二行显示GT, dif_feature[-1], 直方图
    """
    # 只取最后一个特征图（通常是最小的，比如 8×8）
    dif_feat = dif_features[-1]  # 假设最后一个是最深层
    
    if dif_feat.is_cuda:
        dif_feat = dif_feat.cpu()
    
    feat = dif_feat[0].detach().numpy()  # [C, H, W]
    
    # 选择激活最强的通道
    channel_activations = np.mean(feat, axis=(1, 2))
    best_channel = np.argmax(channel_activations)
    feature_map = feat[best_channel]
    
    # 归一化到[0, 1]范围
    feature_min = feature_map.min()
    feature_max = feature_map.max()
    if feature_max - feature_min > 1e-8:
        feature_map = (feature_map - feature_min) / (feature_max - feature_min)
    else:
        feature_map = np.zeros_like(feature_map)
    
    # 准备pred和gt的RGB可视化
    pred = pred[0].astype(np.uint8)
    gt = gt[0].astype(np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

    # 红色表示消失，蓝色表示新增
    pred_rgb[pred == 1] = [255, 0, 0]  # 消失 - 红色
    pred_rgb[pred == 2] = [0, 0, 255]  # 新增 - 蓝色
    gt_rgb[gt == 1] = [255, 0, 0]  # 消失 - 红色
    gt_rgb[gt == 2] = [0, 0, 255]  # 新增 - 蓝色

    # 创建2行3列的子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：Image A, Image B, Prediction
    img_A_path = os.path.join(args.data_folder, 'test/A', name[0])
    img_B_path = os.path.join(args.data_folder, 'test/B', name[0])
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    
    axes[0, 0].imshow(img_A)
    axes[0, 0].set_title('Image A (T1)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_B)
    axes[0, 1].set_title('Image B (T2)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_rgb)
    axes[0, 2].set_title('Prediction\nRed: Disappeared, Blue: New', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行：GT, dif_feature[-1], 直方图
    axes[1, 0].imshow(gt_rgb)
    axes[1, 0].set_title('Ground Truth\nRed: Disappeared, Blue: New', fontsize=12)
    axes[1, 0].axis('off')
    
    # dif_feature[-1] 特征图
    im = axes[1, 1].imshow(feature_map, cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
    axes[1, 1].axis('off')
    
    # 直方图
    axes[1, 2].hist(feature_map.flatten(), bins=50, color='red', alpha=0.7, range=(0, 1))
    axes[1, 2].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.visualize_path, f'{name[0]}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def visualize_semantic_maps(semantic_maps, pred, gt, name, args):
    """
    可视化语义增强模块的gate_weights（消失/新增/背景权重）
    
    Args:
        semantic_maps: 包含gate_weights的字典，形状为[B, 3, H, W]
        pred: 预测结果，形状[H, W]
        gt: 真值结果，形状[H, W]
        name: 文件名列表
        args: 包含数据路径和可视化路径的参数
    """
    name = name[0]
    img_A_path = os.path.join(args.data_folder, 'test/A', name)
    img_B_path = os.path.join(args.data_folder, 'test/B', name)
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)

    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

    pred = pred[0].astype(np.uint8)
    gt = gt[0].astype(np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

    pred_rgb[pred == 1] = [255, 0, 0]  # 消失 - 红色
    pred_rgb[pred == 2] = [0, 0, 255]  # 新增 - 蓝色
    gt_rgb[gt == 1] = [255, 0, 0]  # 消失 - 红色
    gt_rgb[gt == 2] = [0, 0, 255]  # 新增 - 蓝色

    # 提取gate_weights
    if 'gate_weights' in semantic_maps:
        gate_weights = semantic_maps['gate_weights']
        if gate_weights.is_cuda:
            gate_weights = gate_weights.cpu()
        gate_weights = gate_weights[0].detach().numpy()  # [3, H, W]
    else:
        print("Warning: No gate_weights found in semantic_maps")
        return

    # 创建2行4列的子图布局
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 第一行：原始数据
    axes[0, 0].imshow(img_A)
    axes[0, 0].set_title('Image A (T1)', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_B)
    axes[0, 1].set_title('Image B (T2)', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_rgb)
    axes[0, 2].set_title('Prediction\nRed: Disappeared, Blue: New', fontsize=12)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(gt_rgb)
    axes[0, 3].set_title('Ground Truth\nRed: Disappeared, Blue: New', fontsize=12)
    axes[0, 3].axis('off')

    # 第二行：语义权重图
    # 消失权重 (gate_weights[0])
    disappear_weight = gate_weights[0]
    im1 = axes[1, 0].imshow(disappear_weight, cmap='RdBu_r', vmin=0, vmax=1)
    axes[1, 0].set_title('Disappearance Weight\n(High=More Likely Disappeared)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)

    # 新增权重 (gate_weights[1])
    appear_weight = gate_weights[1]
    im2 = axes[1, 1].imshow(appear_weight, cmap='RdBu_r', vmin=0, vmax=1)
    axes[1, 1].set_title('Appearance Weight\n(High=More Likely New)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)

    # 背景权重 (gate_weights[2])
    background_weight = gate_weights[2]
    im3 = axes[1, 2].imshow(background_weight, cmap='RdBu_r', vmin=0, vmax=1)
    axes[1, 2].set_title('Background Weight\n(High=No Change)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], shrink=0.8)

    # 第四列留空或显示权重之和
    weight_sum = gate_weights.sum(axis=0)  # 三个权重之和，应该接近1
    im4 = axes[1, 3].imshow(weight_sum, cmap='viridis', vmin=0, vmax=1)
    axes[1, 3].set_title('Sum of All Weights\n(Should be ~1.0)', fontsize=12)
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(args.visualize_path, f'semantic_{name}'), dpi=150, bbox_inches='tight')
    plt.close(fig)