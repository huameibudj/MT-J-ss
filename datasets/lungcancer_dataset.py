import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# 肿瘤（红）和非肿瘤（蓝）标签调色板
IGNORE_INDEX = 255

FIXED_PALETTE = {
    (128, 0, 0): 0,      # 肿瘤区域
    (0, 130, 200): 1,    # 正常区域
    (0, 0, 0): IGNORE_INDEX  # 其他区域 / 背景，不参与训练
}

_warned_once = False  # 控制只输出一次警告信息

def rgb_to_index(label_img, palette, ignore_index=255):
    global _warned_once
    label_np = np.array(label_img)
    index_mask = np.full(label_np.shape[:2], ignore_index, dtype=np.uint8)

    matched = np.zeros(label_np.shape[:2], dtype=bool)

    for rgb, idx in palette.items():
        match = np.all(label_np == rgb, axis=-1)
        index_mask[match] = idx
        matched |= match

    if not np.all(matched) and not _warned_once:
        unknown_count = np.sum(~matched)
        print(
            f"[WARNING] Detected {unknown_count} pixel(s) that do not match known color classes. "
            f"They will be set to ignore_index={ignore_index}."
        )
        _warned_once = True

    return index_mask

class LungCancerDataset(Dataset):
    def __init__(self, image_input, label_input, palette=FIXED_PALETTE):
        """
        image_input: str (directory) or List[str]
        label_input: str (directory) or List[str]
        """
        if isinstance(image_input, str):
            self.image_paths = sorted([
                os.path.join(image_input, f)
                for f in os.listdir(image_input)
                if f.lower().endswith((".png", ".jpg"))
            ])
        else:
            self.image_paths = image_input

        if isinstance(label_input, str):
            self.label_paths = sorted([
                os.path.join(label_input, f)
                for f in os.listdir(label_input)
                if f.lower().endswith((".png", ".jpg"))
            ])
        else:
            self.label_paths = label_input

        assert len(self.image_paths) == len(self.label_paths), "[ERROR] Mismatch between image and label counts."

        self.transform_image = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        self.transform_label = T.Resize((224, 224), interpolation=Image.NEAREST)
        self.palette = palette

        print(f"[INFO] Loaded {len(self.image_paths)} samples.")
        print(f"[INFO] Using fixed color palette (RGB -> index): {self.palette}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("RGB")

        image = self.transform_image(image)
        label = self.transform_label(label)
        label_index = rgb_to_index(label, self.palette, ignore_index=255)
        label_tensor = torch.from_numpy(label_index).long()

        return image, label_tensor
