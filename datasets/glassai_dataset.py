import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# 固定的颜色调色板（RGB -> 类别索引）
FIXED_PALETTE = {
    (255, 255, 255): 0,  # 背景
    (255, 255, 0):   1,#黄色
    (255, 0, 0):     2,#红色
    (0, 255, 255):   3,#青色
    (0, 0, 255):     4,#蓝色
    (0, 255, 0):     5# 绿色
}

_warned_once = False  # 控制只输出一次警告信息


def rgb_to_index(label_img, palette, ignore_index=255):
    """
    RGB -> index mask
    未匹配到 palette 的像素设为 ignore_index（推荐 255）
    """
    global _warned_once

    label_np = np.array(label_img)  # (H,W,3)
    h, w = label_np.shape[:2]

    # 默认填 ignore_index，避免未知颜色被当成背景
    index_mask = np.full((h, w), fill_value=int(ignore_index), dtype=np.uint8)

    matched = np.zeros((h, w), dtype=bool)
    for rgb, idx in palette.items():
        match = np.all(label_np == rgb, axis=-1)
        index_mask[match] = int(idx)
        matched |= match

    if (not np.all(matched)) and (not _warned_once):
        unknown_count = int(np.sum(~matched))
        print(f"[WARNING] 检测到 {unknown_count} 个像素不匹配已知颜色，将设为 ignore_index={ignore_index}。（仅提示一次）")
        _warned_once = True

    return index_mask


def _scan_dir_as_stem_map(folder: str, exts=(".png", ".jpg", ".jpeg")):
    """
    扫描目录，返回：{stem: full_path}
    stem = 去掉扩展名的文件名，例如 'img_0001'
    """
    m = {}
    for f in os.listdir(folder):
        if f.lower().endswith(exts):
            stem = os.path.splitext(f)[0]
            m[stem] = os.path.join(folder, f)
    return m


class GlassAIDataset(Dataset):
    def __init__(
        self,
        image_paths=None,
        label_paths=None,
        image_dir=None,
        label_dir=None,
        transform=None,
        palette=FIXED_PALETTE,
        resize_hw=None,          # (224,224) or None
        ignore_index=255,
        strict_match=True        # ✅ 新增：是否严格按文件名匹配（强烈建议 True）
    ):
        """
        支持两种方式：
        - 显式列表：image_paths + label_paths（认为已对齐）
        - 目录扫描：image_dir + label_dir（按文件名 stem 严格匹配，避免错配）
        """
        self.palette = palette
        self.ignore_index = int(ignore_index)

        # --------------------------
        # 1) 构建 image_paths / label_paths
        # --------------------------
        if image_paths is not None and label_paths is not None:
            if len(image_paths) != len(label_paths):
                raise AssertionError("[ERROR] image_paths 与 label_paths 数量不一致。")
            self.image_paths = list(image_paths)
            self.label_paths = list(label_paths)

        elif image_dir is not None and label_dir is not None:
            img_map = _scan_dir_as_stem_map(image_dir)
            lbl_map = _scan_dir_as_stem_map(label_dir)

            img_keys = set(img_map.keys())
            lbl_keys = set(lbl_map.keys())
            common = sorted(list(img_keys & lbl_keys))

            if len(common) == 0:
                raise RuntimeError(
                    f"[ERROR] 在目录中未找到可匹配的图像/标签文件名。\n"
                    f"image_dir={image_dir}\nlabel_dir={label_dir}"
                )

            # 统计缺失项（方便你发现数据问题）
            missing_lbl = sorted(list(img_keys - lbl_keys))
            missing_img = sorted(list(lbl_keys - img_keys))

            if strict_match:
                # 严格模式：只用交集（最安全）
                self.image_paths = [img_map[k] for k in common]
                self.label_paths = [lbl_map[k] for k in common]
            else:
                # 非严格：仍然只用交集（为了安全这里也不建议做别的）
                self.image_paths = [img_map[k] for k in common]
                self.label_paths = [lbl_map[k] for k in common]

            print(f"[INFO] 扫描目录匹配完成：matched={len(common)}")
            if len(missing_lbl) > 0:
                print(f"[WARNING] 有 {len(missing_lbl)} 张图像找不到对应标签（仅展示前 5 个）：{missing_lbl[:5]}")
            if len(missing_img) > 0:
                print(f"[WARNING] 有 {len(missing_img)} 张标签找不到对应图像（仅展示前 5 个）：{missing_img[:5]}")

        else:
            raise ValueError("请提供 (image_paths, label_paths) 或 (image_dir, label_dir)")

        # --------------------------
        # 2) 图像 transform
        # --------------------------
        if transform is not None:
            self.transform_image = transform
        else:
            if resize_hw is not None:
                self.transform_image = T.Compose([T.Resize(resize_hw), T.ToTensor()])
            else:
                self.transform_image = T.ToTensor()

        # --------------------------
        # 3) 标签 resize（NEAREST 避免插值颜色污染）
        # --------------------------
        self.resize_label = None
        if resize_hw is not None:
            self.resize_label = T.Resize(resize_hw, interpolation=Image.NEAREST)

        # --------------------------
        # 4) 打印信息
        # --------------------------
        print(f"[INFO] Loaded {len(self.image_paths)} samples.")
        print(f"[INFO] ignore_index={self.ignore_index}")
        print(f"[INFO] Using palette (RGB -> index): {self.palette}")
        if resize_hw is not None:
            print(f"[INFO] Resize to: {resize_hw}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("RGB")

        if self.resize_label is not None:
            label = self.resize_label(label)

        image = self.transform_image(image)
        label_index = rgb_to_index(label, self.palette, ignore_index=self.ignore_index)
        label_tensor = torch.from_numpy(label_index).long()

        return image, label_tensor
