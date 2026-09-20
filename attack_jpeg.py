#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击方法 1：JPEG 重编码（JPEG Re-encode）

原理：
    DeepFake 图像在生成过程中（换脸融合）会在融合边界留下高频振铃、重采样伪影
    与局部纹理异常。这些取证痕迹在 8×8 分块 DCT 的高频系数上表现突出，因此频域
    类检测器（SRM 噪声残差 + 分块 DCT + FFT 高频带）能有效区分真伪。

    JPEG 以 8×8 分块 DCT + 量化表丢弃高频系数实现有损压缩。对伪造图重新执行
    "编码 -> 解码" 会：
      1. 打乱原图的分块边界（原 DCT 能量分布被重排）；
      2. 量化丢弃高频取证痕迹（融合边界的振铃、局部纹理异常被抹平）；
    从而削弱依赖高频/频域统计的检测器。质量越低压缩越强、隐匿越明显，但失真越大。

接口（平台契约）：
    attack(sample) -> {"image": RGB uint8 H×W×3，与输入同尺寸}
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

# JPEG 质量池（赛题 A0-1：quality ∈ {55, 65, 75, 85}）
JPEG_QUALITIES = (55, 65, 75, 85)


class JPEGReencodeAttack:
    """JPEG 重编码攻击（交付 Baseline 的最小攻击示例）。"""

    name = "jpeg"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        if quality not in JPEG_QUALITIES:
            quality = int(min(JPEG_QUALITIES, key=lambda q: abs(q - quality)))
        self.quality = quality
        self.seed = seed

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(np.asarray(image, dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def attack(sample, quality: int = 85, seed: int = 42) -> dict:
    """JPEG 重编码攻击入口。"""
    image = sample["image"]
    out = JPEGReencodeAttack(quality=quality, seed=seed).apply(image)
    return {"image": out}
