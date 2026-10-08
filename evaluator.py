import numpy as np

class Evaluator:
    """变化检测评估器（3类：背景/消失/新增）"""
    def __init__(self, num_class=3):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
    
    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
    
    def add_batch(self, gt, pred):
        """
        gt:   [H, W] or [B, H, W]
        pred: [H, W] or [B, H, W]
        """
        # 展平为一维
        gt = gt.flatten()
        pred = pred.flatten()
        
        # 过滤无效标签
        valid_mask = (gt >= 0) & (gt < self.num_class)
        gt = gt[valid_mask]
        pred = pred[valid_mask]
        
        # 手动构建混淆矩阵
        for g, p in zip(gt, pred):
            self.confusion_matrix[g, p] += 1
    
    def Pixel_Accuracy(self):
        """像素准确率（PA）"""
        acc = np.diag(self.confusion_matrix).sum() / max(self.confusion_matrix.sum(), 1e-8)
        return acc
    
    def Pixel_Accuracy_Class(self):
        """每个类别的像素准确率"""
        acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-8)
        return acc
    
    def Mean_Intersection_over_Union(self):
        """平均交并比（MIoU）"""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )
        iou = intersection / (union + 1e-8)
        miou = np.nanmean(iou)
        return miou
    
    def Frequency_Weighted_Intersection_over_Union(self):
        """频率加权交并比（FWIoU）"""
        freq = self.confusion_matrix.sum(axis=1) / max(self.confusion_matrix.sum(), 1e-8)
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )
        iou = intersection / (union + 1e-8)
        fw_iou = (freq * iou).sum()
        return fw_iou
    
    def get_scores(self):
        """返回所有指标"""
        pa = self.Pixel_Accuracy()
        pa_class = self.Pixel_Accuracy_Class()
        miou = self.Mean_Intersection_over_Union()
        fw_iou = self.Frequency_Weighted_Intersection_over_Union()
        
        return {
            'Pixel_Accuracy': pa,
            'Pixel_Accuracy_Class': pa_class,
            'Mean_IoU': miou,
            'Freq_W_IoU': fw_iou
        }