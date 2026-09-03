import os
import json
import random
from pathlib import Path
from PIL import Image, ImageFilter

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _as_path_str(x):
    return str(x).replace("\\", os.sep)


def _resolve_path(path_value, *, split_file_dir=None, data_root=None, image_dir=None, label_dir=None, kind="image"):
    if path_value is None:
        return None

    p_raw = _as_path_str(path_value).strip()
    p = Path(p_raw)

    if p.is_absolute():
        return str(p)

    base_dir = image_dir if kind == "image" else label_dir

    candidates = []
    if split_file_dir is not None:
        candidates.append(Path(split_file_dir) / p_raw)
    if data_root is not None:
        candidates.append(Path(_as_path_str(data_root)) / p_raw)
    if base_dir is not None:
        candidates.append(Path(_as_path_str(base_dir)) / p_raw)

    for c in candidates:
        if c.exists():
            return str(c)

    # split.json 里只有样本名 / stem，例如 P2_F9_4_12_8
    if base_dir is not None and Path(p_raw).suffix == "":
        b = Path(_as_path_str(base_dir))

        for ext in IMG_EXTS:
            c = b / f"{p_raw}{ext}"
            if c.exists():
                return str(c)

        try:
            matches = []
            for ext in IMG_EXTS:
                matches.extend(b.rglob(f"{p_raw}{ext}"))
            if matches:
                return str(sorted(matches)[0])
        except Exception:
            pass

    # 有后缀但可能在子目录里
    if base_dir is not None and Path(p_raw).suffix != "":
        b = Path(_as_path_str(base_dir))
        try:
            matches = list(b.rglob(Path(p_raw).name))
            if matches:
                return str(sorted(matches)[0])
        except Exception:
            pass

    if base_dir is not None:
        if Path(p_raw).suffix == "":
            return str(Path(_as_path_str(base_dir)) / f"{p_raw}.png")
        return str(Path(_as_path_str(base_dir)) / p_raw)

    if candidates:
        return str(candidates[0])

    return p_raw


def _get_first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def _infer_mask_from_image(image_path, *, label_dir=None, mask_suffix=""):
    if image_path is None or label_dir is None:
        return None

    img = Path(image_path)
    stem = img.stem + str(mask_suffix or "")
    b = Path(_as_path_str(label_dir))

    for ext in IMG_EXTS:
        c = b / f"{stem}{ext}"
        if c.exists():
            return str(c)

    try:
        matches = []
        for ext in IMG_EXTS:
            matches.extend(b.rglob(f"{stem}{ext}"))
        if matches:
            return str(sorted(matches)[0])
    except Exception:
        pass

    return str(b / f"{stem}{img.suffix or '.png'}")


def load_split_json_paths(
    split_file,
    split,
    *,
    data_root=None,
    image_dir=None,
    label_dir=None,
    mask_suffix="",
):
    split_file = Path(_as_path_str(split_file))
    split_file_dir = split_file.parent

    with open(split_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "splits" in obj and isinstance(obj["splits"], dict):
        obj = obj["splits"]

    if not isinstance(obj, dict) or split not in obj:
        raise KeyError(
            f"[ERROR] split_file={split_file} 中找不到 split='{split}'。"
            f"需要包含 train/val/test，或 splits.train/splits.val/splits.test。"
        )

    items = obj[split]
    if not isinstance(items, list):
        raise TypeError(f"[ERROR] split_file 中 {split} 必须是 list，但得到: {type(items)}")

    image_paths = []
    label_paths = []

    for idx, item in enumerate(items):
        image_value = None
        label_value = None

        if isinstance(item, dict):
            image_value = _get_first(
                item,
                ["image", "img", "image_path", "img_path", "image_file", "file", "filename", "name", "id"],
            )
            label_value = _get_first(
                item,
                ["mask", "label", "label_path", "mask_path", "mask_file", "label_file", "gt", "target"],
            )

        elif isinstance(item, (list, tuple)):
            if len(item) >= 2:
                image_value, label_value = item[0], item[1]
            elif len(item) == 1:
                image_value = item[0]

        elif isinstance(item, str):
            image_value = item

        else:
            raise TypeError(f"[ERROR] split_file 中 {split}[{idx}] 格式不支持: {type(item)}")

        image_path = _resolve_path(
            image_value,
            split_file_dir=split_file_dir,
            data_root=data_root,
            image_dir=image_dir,
            kind="image",
        )

        if label_value is not None:
            label_path = _resolve_path(
                label_value,
                split_file_dir=split_file_dir,
                data_root=data_root,
                label_dir=label_dir,
                kind="label",
            )
        else:
            label_path = _infer_mask_from_image(
                image_path,
                label_dir=label_dir,
                mask_suffix=mask_suffix,
            )

        if image_path is None or label_path is None:
            raise ValueError(f"[ERROR] 无法解析 {split}[{idx}] 的 image/mask: {item}")

        image_paths.append(str(image_path))
        label_paths.append(str(label_path))

    return image_paths, label_paths


class GlasDataset(Dataset):
    def __init__(
        self,
        image_dir=None,
        label_dir=None,
        resize_hw=None,
        ignore_index=255,
        augment=False,
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotate90_prob=0.5,
        color_jitter_prob=0.5,
        blur_prob=0.15,
        image_paths=None,
        label_paths=None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir

        if image_paths is not None or label_paths is not None:
            if image_paths is None or label_paths is None:
                raise ValueError("[ERROR] image_paths 和 label_paths 必须同时提供")
            self.image_paths = [str(x) for x in image_paths]
            self.label_paths = [str(x) for x in label_paths]
        else:
            if image_dir is None or label_dir is None:
                raise ValueError("[ERROR] 未提供 image_paths/label_paths 时必须提供 image_dir/label_dir")

            self.image_paths = sorted([
                os.path.join(image_dir, x)
                for x in os.listdir(image_dir)
                if x.lower().endswith(IMG_EXTS)
            ])
            self.label_paths = sorted([
                os.path.join(label_dir, x)
                for x in os.listdir(label_dir)
                if x.lower().endswith(IMG_EXTS)
            ])

        if len(self.image_paths) != len(self.label_paths):
            raise RuntimeError(
                f"[ERROR] image 和 mask 数量不一致: "
                f"{len(self.image_paths)} images vs {len(self.label_paths)} masks"
            )

        if len(self.image_paths) == 0:
            raise RuntimeError("[ERROR] Dataset 为空，请检查 data_root/image_dir/label_dir/split_file 配置")

        self.resize_hw = self._parse_resize_hw(resize_hw)
        self.ignore_index = int(ignore_index)

        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate90_prob = float(rotate90_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.blur_prob = float(blur_prob)

        self.img_tf = T.ToTensor()

    def _parse_resize_hw(self, resize_hw):
        if resize_hw is None:
            return None

        if isinstance(resize_hw, (list, tuple)) and len(resize_hw) == 2:
            h, w = int(resize_hw[0]), int(resize_hw[1])
            return (h, w)

        raise ValueError(f"[ERROR] resize_hw 应该是 None 或 [H, W]，但得到: {resize_hw}")

    def __len__(self):
        return len(self.image_paths)

    def _resize_pair(self, img, mask):
        if self.resize_hw is None:
            return img, mask

        h, w = self.resize_hw
        img = img.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)

        return img, mask

    def _apply_train_aug(self, img, mask):
        if random.random() < self.hflip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() < self.vflip_prob:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if random.random() < self.rotate90_prob:
            k = random.randint(1, 3)
            angle = 90 * k

            img = TF.rotate(
                img,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0,
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                expand=False,
                fill=0,
            )

        if random.random() < self.color_jitter_prob:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.85, 1.15)

            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)

        if random.random() < self.blur_prob:
            radius = random.uniform(0.1, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img, mask

    def _mask_to_index(self, mask):
        mask = np.array(mask)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # binary mask:
        # 0 -> background
        # >0 -> foreground / gland
        mask = (mask > 0).astype(np.int64)

        return torch.from_numpy(mask).long()

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")
        mask = Image.open(self.label_paths[i])

        img, mask = self._resize_pair(img, mask)

        if self.augment:
            img, mask = self._apply_train_aug(img, mask)

        img = self.img_tf(img)
        mask = self._mask_to_index(mask)

        return img, mask