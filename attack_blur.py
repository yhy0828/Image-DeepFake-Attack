#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击方法 3：模糊-锐化（Blur & Sharpen）

原理：
    DeepFake 检测器的频域特征大量依赖 SRM 高通噪声残差（本质是对图像做高通滤波，
    提取边缘/噪声），空域像素特征则依赖水平/垂直梯度统计。伪造图像的融合边界
    振铃与内部纹理噪声会使这些高通响应异常偏高。

    本攻击采用"先模糊后锐化"：
      1. 轻度高斯模糊（σ≈1.2）先抑制高频噪声与振铃，降低 SRM 残差能量与梯度；
      2. 再用 UnsharpMask 重新锐化，恢复主体边缘的视觉清晰度（避免整体变"糊"）。
    这样在降低高频取证痕迹的同时，保持人脸五官/轮廓的主观清晰度，兼顾保真度。

接口（平台契约）：
    attack(sample) -> {"image": RGB uint8 H×W×3，与输入同尺寸}
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


class JPEGReencodeAttack:
    """模糊-锐化攻击。"""

    name = "blur_sharpen"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        self.blur_radius = 1.2
        self.sharpen_factor = 1.5
        self.seed = int(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(np.asarray(image, dtype=np.uint8))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        radius_i = max(1, int(round(self.blur_radius)))
        amount = max(0.0, self.sharpen_factor * 100.0)
        sharpened = blurred.filter(ImageFilter.UnsharpMask(
            radius=radius_i, percent=int(round(amount)), threshold=0))
        return np.asarray(sharpened.convert("RGB"), dtype=np.uint8)


def attack(sample, quality: int = 85, seed: int = 42) -> dict:
    """模糊-锐化攻击入口。"""
    image = sample["image"]
    out = JPEGReencodeAttack(quality=quality, seed=seed).apply(image)
    return {"image": out}
