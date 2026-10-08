import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import copy
import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
# import cv2 as cv
from imageio import imread
from PIL import Image
from random import *

# Transformer Decoder
class WHUBSDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split, token_folder=None, vocab_file=None, max_length=41, allow_unk=0,
                 max_iters=None):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
        if not max_iters == None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]
        self.files = []
        if split == 'train':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split + '/A/' + name.split('-')[0])
                img_fileB = os.path.join(data_folder + '/' + split + '/B/' + name.split('-')[0])
                img_label = os.path.join(data_folder + '/' + split + '/label3/' + name.split('-')[0])
                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_label)

                if '-' in name:
                    token_id = name.split('-')[-1]
                else:
                    token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('-')[0]
                })
        elif split == 'val':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split + '/A/' + name)
                img_fileB = os.path.join(data_folder + '/' + split + '/B/' + name)
                img_label = os.path.join(data_folder + '/' + split + '/label3/' + name)
                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_label)

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
        elif split == 'test':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split + '/A/' + name)
                img_fileB = os.path.join(data_folder + '/' + split + '/B/' + name)
                img_label = os.path.join(data_folder + '/' + split + '/label3/' + name)
                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_label)

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = datafiles["imgA"]
        imgB = datafiles["imgB"]
        seg_label = datafiles["seg_label"]

        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)

        # ========== 核心修改：三分类标签处理 ==========
        seg_label = np.asarray(seg_label, dtype=np.uint8)

        # 1. 如果是单通道，直接转为 int64（假设已经是 0,1,2 的索引图）
        if seg_label.ndim == 2:
            seg_label = seg_label.astype(np.int64)
            # 安全检查：如果值域异常，报错提示
            unique_vals = np.unique(seg_label)
            if not np.all(np.isin(unique_vals, [0, 1, 2])):
                raise ValueError(f"单通道标签值异常: {unique_vals}，期望 [0,1,2]")

        # 2. 如果是3通道（RGB），进行颜色到类别的映射
        elif seg_label.ndim == 3 and seg_label.shape[2] == 3:
            h, w, _ = seg_label.shape
            label_map = np.zeros((h, w), dtype=np.int64)

            # 颜色映射规则（基于 RGB 空间）
            # 注意：imageio.imread 通常返回 RGB 顺序
            r, g, b = seg_label[..., 0], seg_label[..., 1], seg_label[..., 2]

            # 红色（消失）：R 高，G/B 低
            red_mask = (r > 200) & (g < 50) & (b < 50)
            # 蓝色（新增）：B 高，R/G 低
            blue_mask = (b > 200) & (r < 50) & (g < 50)
            # 黑色（背景）：R/G/B 都低（默认 0，无需赋值）

            label_map[red_mask] = 1  # 消失 -> 1
            label_map[blue_mask] = 2  # 新增 -> 2
            seg_label = label_map

        else:
            # 其他维度，强制转为 2D
            seg_label = seg_label.reshape(seg_label.shape[0], seg_label.shape[1]).astype(np.int64)

        # 归一化：减去均值，除以标准差
        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

            # ========== Token 处理（保持原样） ==========
        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
                if self.split == 'train' and nochange_cap in caption_list:
                    tokens = nochange_cap
                tokens_encode = encode(tokens, self.word_vocab,
                                       allow_unk=self.allow_unk == 1)
                token_all[j, :len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        return imgA.copy(), imgB.copy(), seg_label.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(
            token_len), name

def blip_caption_preprocess(caption, max_words=50):
    """BLIP caption 预处理函数（替代 BlipCaptionProcessor）"""
    caption = caption.lower()
    caption = re.sub(r'([.!\"()*#:;~])', ' ', caption)
    caption = re.sub(r'\s{2,}', ' ', caption)
    caption = caption.strip()
    
    words = caption.split()
    if len(words) > max_words:
        caption = ' '.join(words[:max_words])
    
    return caption


# ============================================================
# Dataset
# ============================================================

class WHUBSCaptionDataset(Dataset):
    """
    将 WHUBS 变化检测数据集转换为 Caption 风格数据集
    输出格式与 CaptionDataset 完全一致
    """

    def __init__(
        self,
        img_folder,
        data_folder,
        split,
        max_words=50
    ):
        """
        Args:
            data_folder: WHU-BSCD 根目录
            img_folder: 图像文件夹
            split: 'train' / 'val' / 'test'
        """
        self.data_folder = data_folder
        self.img_folder = img_folder
        self.split = split
        self.max_words = max_words
        
        assert self.split in {'train', 'val', 'test'}, f"split must be 'train', 'val' or 'test', got '{split}'"

        # 加载 conversation列表
        self.annotation = []
        
        json_path = os.path.join(data_folder, f"{split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        print(f"Loading annotations from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for d in data:
            dialog = copy.deepcopy(d)
            all_turns = dialog.get('conversations', [])
            
            # 拆分成 (question, answer) 对
            all_turns = [
                {
                    "answer": all_turns[d+1]['value'],
                    "question": all_turns[d]['value'],
                }
                for d in range(0, len(all_turns), 2)
            ]
            
            for i in range(len(all_turns)):
                dialog_instance = copy.deepcopy(dialog)
                dialogue_context = ' '.join([f" q: {t['question']} a: {t['answer']}" 
                                            for t in all_turns[:i]]).strip()
                last_turn = all_turns[i]

                question = last_turn["question"]
                answer = last_turn["answer"]
                
                if dialogue_context == '':
                    dialog_instance["question"] = 'q: ' + question
                else:
                    dialog_instance["question"] = dialogue_context + '  q: ' + question
                
                dialog_instance["answer"] = answer
                dialog_instance['conversations'] = ''
                
                self.annotation.append(dialog_instance)
        
        print(f"Loaded {len(self.annotation)} samples for split '{split}'")

        # 图像归一化参数（S2Looking 原始设定）
        # 注意：这里已经乘以255，所以后续不需要再乘
        self.mean = np.array([0.39073 * 255, 0.38623 * 255, 0.32989 * 255], dtype=np.float32)
        self.std = np.array([0.15329 * 255, 0.14628 * 255, 0.13648 * 255], dtype=np.float32)

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        # ---------- 读取图像 ----------
        img_paths = ann.get("image", [])
        if len(img_paths) < 3:
            raise ValueError(f"Sample {index} has invalid image paths: {img_paths}")
        
        imgA_path = os.path.join(self.img_folder, img_paths[0])
        imgB_path = os.path.join(self.img_folder, img_paths[1])
        label_path = os.path.join(self.img_folder, img_paths[2])
        
        # 检查文件是否存在
        for path in [imgA_path, imgB_path, label_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            imgA = imread(imgA_path)
            imgB = imread(imgB_path)
            seg_label = imread(label_path)
        except Exception as e:
            raise RuntimeError(f"Error loading images for sample {index}: {e}")
        
        # 转换为 float32
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        
        # HWC → CHW
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        
        # 标签处理
        seg_label = self._process_label(seg_label)
        
        # 归一化（注意：mean/std 已经乘过255）
        for i in range(3):
            imgA[i] = (imgA[i] - self.mean[i]) / self.std[i]
            imgB[i] = (imgB[i] - self.mean[i]) / self.std[i]

        # ---------- 读取文本描述 ----------
        text_input = ann.get('question', '').replace('<image>', '').strip()
        text_output = ann.get('answer', '')
        
        # 使用 BLIP caption 预处理
        text_input = blip_caption_preprocess(text_input, max_words=self.max_words)
        text_output = blip_caption_preprocess(text_output, max_words=self.max_words)
        
        return (
            imgA.copy(), 
            imgB.copy(), 
            seg_label.copy(), 
            text_input, 
            text_output, 
            ann.get("id", index), 
            ann.get("changeflag", False)
        )
    
    def _process_label(self, seg_label):
        """处理分割标签"""
        # 1. 如果是单通道，直接转为 int64
        if seg_label.ndim == 2:
            seg_label = seg_label.astype(np.int64)
            unique_vals = np.unique(seg_label)
            if not np.all(np.isin(unique_vals, [0, 1, 2])):
                raise ValueError(f"单通道标签值异常: {unique_vals}，期望 [0,1,2]")
            return seg_label
        
        # 2. 如果是3通道（RGB），进行颜色到类别的映射
        elif seg_label.ndim == 3 and seg_label.shape[2] == 3:
            h, w, _ = seg_label.shape
            label_map = np.zeros((h, w), dtype=np.int64)

            # 颜色映射规则
            r, g, b = seg_label[..., 0], seg_label[..., 1], seg_label[..., 2]

            # 红色（消失）：R 高，G/B 低
            red_mask = (r > 200) & (g < 50) & (b < 50)
            # 蓝色（新增）：B 高，R/G 低
            blue_mask = (b > 200) & (r < 50) & (g < 50)

            label_map[red_mask] = 1  # 消失 -> 1
            label_map[blue_mask] = 2  # 新增 -> 2
            
            # 检查是否有未分类的像素
            unclassified = (label_map == 0) & (r > 50) & (g > 50) & (b > 50)
            if np.any(unclassified):
                print(f"Warning: Found {np.sum(unclassified)} pixels with unknown colors")
            
            return label_map
        
        else:
            # 其他维度，强制转为 2D
            return seg_label.reshape(seg_label.shape[0], seg_label.shape[1]).astype(np.int64)


# ============================================================
# 测试脚本
# ============================================================

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", default="../../WHU-BSCD-dataset/images")
    parser.add_argument("--data_folder", default="../../WHU-BSCD")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("[1] 初始化 Dataset")
    print("=" * 60)
    
    try:
        dataset = WHUBSCaptionDataset(
            img_folder=args.img_folder,
            data_folder=args.data_folder,
            split=args.split,
        )
        print(f"  样本总数: {len(dataset)}")
        assert len(dataset) > 0, "数据集为空, 请检查路径和 split 文件!"
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        exit(1)

    print("\n" + "=" * 60)
    print("[2] 测试 __getitem__ (前 3 条)")
    print("=" * 60)
    
    for i in range(min(3, len(dataset))):
        try:
            imgA, imgB, seg_label, text_input, text_output, sample_id, changeflag = dataset[i]
            
            print(f"\n--- 样本 {i} (ID: {sample_id}) ---")
            print(f"  imgA: shape={imgA.shape}, dtype={imgA.dtype}, "
                  f"range=[{imgA.min():.2f}, {imgA.max():.2f}]")
            print(f"  imgB: shape={imgB.shape}, dtype={imgB.dtype}, "
                  f"range=[{imgB.min():.2f}, {imgB.max():.2f}]")
            print(f"  label: shape={seg_label.shape}, dtype={seg_label.dtype}, "
                  f"unique={np.unique(seg_label)}")
            print(f"  text_input: {text_input}")
            print(f"  text_output: {text_output}")
            print(f"  changeflag: {changeflag}")
        except Exception as e:
            print(f"❌ 样本 {i} 读取失败: {e}")

    print("\n" + "=" * 60)
    print(f"[3] 测试 DataLoader (batch_size={args.batch_size})")
    print("=" * 60)
    
    try:
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.num_workers,
            collate_fn=lambda batch: batch,
        )
        batch = next(iter(loader))
        print(f"  batch 长度: {len(batch)}")
        
        for i, item in enumerate(batch):
            imgA, imgB, seg_label, text_input, text_output, sample_id, changeflag = item
            print(f"\n  --- batch item {i} (ID: {sample_id}) ---")
            print(f"    imgA: shape={imgA.shape}, dtype={imgA.dtype}")
            print(f"    imgB: shape={imgB.shape}, dtype={imgB.dtype}")
            print(f"    label: shape={seg_label.shape}, dtype={seg_label.dtype}")
            print(f"    text_input: {text_input}")
            print(f"    text_output: {text_output}")
    except Exception as e:
        print(f"❌ DataLoader 测试失败: {e}")
    exit(0)
    print("\n" + "=" * 60)
    print("[4] 标签类别分布检查")
    print("=" * 60)
    
    stat = Counter()
    error_samples = []
    
    for i in range(len(dataset)):
        try:
            _, _, seg_label, _, _, sample_id, _ = dataset[i]
            for u in np.unique(seg_label).tolist():
                stat[int(u)] += 1
        except Exception as e:
            error_samples.append((i, str(e)))
            if len(error_samples) > 5:  # 只记录前5个错误
                break
    
    print(f"  类别出现统计 (类别: 像素数): {dict(stat)}")
    print(f"  ✓ 0=背景(不变), 1=消失, 2=新增")
    
    if error_samples:
        print(f"\n⚠️ 发现 {len(error_samples)} 个错误样本:")
        for idx, err in error_samples[:3]:
            print(f"  样本 {idx}: {err}")

    print("\n" + "=" * 60)
    print("[5] 数据统计信息")
    print("=" * 60)
    
    # 随机采样统计
    random_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    text_lengths = []
    
    for idx in random_indices:
        _, _, _, text_input, text_output, _, _ = dataset[idx]
        text_lengths.append(len(text_input.split()))
        text_lengths.append(len(text_output.split()))
    
    if text_lengths:
        print(f"  文本长度统计 (words):")
        print(f"    平均: {np.mean(text_lengths):.1f}")
        print(f"    最小: {np.min(text_lengths)}")
        print(f"    最大: {np.max(text_lengths)}")
        print(f"    中位数: {np.median(text_lengths):.1f}")

    print("\n✅ 数据集构建与读取测试完成!")