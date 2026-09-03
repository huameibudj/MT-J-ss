import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import torchvision.transforms as T

# === 血细胞调色板：RGB → 类别索引 ===
FIXED_PALETTE = {
    (255, 255, 255): 0,  # 背景
    (255, 0, 0): 1,      # 白细胞
    (0, 255, 0): 2,      # 红细胞
    (0, 0, 255): 3       # 血小板
}

_warned_once = False  # 全局警告标志位（只输出一次）

# === 将 RGB 标签图转换为索引图 ===
def rgb_to_index(label_img, palette):
    global _warned_once
    label_np = np.array(label_img)
    index_mask = np.zeros(label_np.shape[:2], dtype=np.uint8)

    matched = np.zeros(label_np.shape[:2], dtype=bool)
    for rgb, idx in palette.items():
        match = np.all(label_np == rgb, axis=-1)
        index_mask[match] = idx
        matched |= match

    if not np.all(matched) and not _warned_once:
        unknown_pixels = label_np[~matched]
        unique_colors, counts = np.unique(unknown_pixels.reshape(-1, 3), axis=0, return_counts=True)

        print("[WARNING] Top unknown colors:")
        for color, count in zip(unique_colors[:5], counts[:5]):
            print(f"  Color {tuple(color)}: {count} pixels")
        print(f"[WARNING] Detected {np.sum(~matched)} unknown pixels in mask.")
        _warned_once = True

    return index_mask

# === 血细胞图像分割数据集 ===
class BloodCellDataset(TorchDataset):
    def __init__(self, image_paths, label_paths, palette=FIXED_PALETTE,
                 resize=True, target_size=(224, 224)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.palette = palette
        self.resize = resize
        self.target_size = target_size

        assert len(self.image_paths) == len(self.label_paths), "[ERROR] 图像与标签数量不一致"
        print(f"[INFO] Loaded {len(self.image_paths)} samples.")
        print(f"[INFO] Using palette: {self.palette}")
        print(f"[INFO] Resize: {self.resize}, Target size: {self.target_size}")

        # 图像变换
        self.image_transform = T.Compose([
            T.Resize(self.target_size) if self.resize else T.Lambda(lambda x: x),
            T.ToTensor()
        ])

        # 标签变换（最近邻插值）
        self.label_transform = T.Resize(self.target_size, interpolation=Image.NEAREST) if self.resize else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        if self.label_transform:
            label = self.label_transform(label)

        image = self.image_transform(image)
        label_index = rgb_to_index(label, self.palette)
        label_tensor = torch.from_numpy(label_index).long()

        return image, label_tensor
