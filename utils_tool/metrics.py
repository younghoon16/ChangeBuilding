import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Recall_Class(self):
        Recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Recall = np.nanmean(Recall)
        return Recall

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU, f'{IoU[0]}   {IoU[1]}  {IoU[2]}'

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # ========== 新增：按你的 IoU 风格返回三个值 ==========
    def Precision_Class_Three(self):
        """返回三个类别的 Precision [P0, P1, P2] 的元组"""
        tp = np.diag(self.confusion_matrix)
        tp_plus_fp = np.sum(self.confusion_matrix, axis=0)  # 列和 = TP + FP
        # 使用 np.divide 防止除零，与你的 IoU 逻辑一致
        precision = np.divide(tp, tp_plus_fp, out=np.zeros_like(tp, dtype=float), where=tp_plus_fp != 0)
        return precision[0], precision[1], precision[2]

    def Recall_Class_Three(self):
        """返回三个类别的 Recall [R0, R1, R2] 的元组"""
        tp = np.diag(self.confusion_matrix)
        tp_plus_fn = np.sum(self.confusion_matrix, axis=1)  # 行和 = TP + FN
        recall = np.divide(tp, tp_plus_fn, out=np.zeros_like(tp, dtype=float), where=tp_plus_fn != 0)
        return recall[0], recall[1], recall[2]

    def F1_Score_Class_Three(self):
        """返回三个类别的 F1 [F1_0, F1_1, F1_2] 的元组"""
        p0, p1, p2 = self.Precision_Class_Three()
        r0, r1, r2 = self.Recall_Class_Three()

        def _safe_f1(p, r):
            if p + r == 0:
                return 0.0
            return 2 * p * r / (p + r)

        return _safe_f1(p0, r0), _safe_f1(p1, r1), _safe_f1(p2, r2)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)