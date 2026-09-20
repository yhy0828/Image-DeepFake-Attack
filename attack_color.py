#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击方法 4：颜色洗白（Color Laundering）

原理：
    换脸 DeepFake 常在被换脸区域引入轻微色差（肤色/光照不一致），检测器的像素统计
    特征（每通道均值/标准差/极差/分位差）与颜色统计特征会捕捉到这种颜色分布偏移。

    本攻击通过轻微的颜色空间非线性变换"洗白"这种色差：
      1. Gamma 校正（γ≈0.9）：改变整体亮度响应曲线，压缩/拉伸中间调，抹平局部色偏；
      2. 饱和度调整（s≈1.1）：把像素向灰度方向或远离灰度方向拉伸，弱化肤色纹理差异；
      3. 色温偏移（t≈0，R 通道 +t、B 通道 -t）：模拟白平衡微调。
    三者叠加使被换脸区域与原图的颜色统计趋于一致，从而降低颜色/像素统计类检测器的
    置信度。由于变换幅度轻微，SSIM/PSNR 保真度保持得很好。

接口（平台契约）：
    attack(sample) -> {"image": RGB uint8 H×W×3，与输入同尺寸}
"""

from __future__ import annotations

import numpy as np


class JPEGReencodeAttack:
    """颜色洗白攻击。"""

    name = "color_laundering"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        self.gamma = 0.9
        self.saturation = 1.1
        self.temperature = 0.0
        self.seed = int(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32) / 255.0
        # 1) Gamma 校正
        img = np.clip(img, 0.0, 1.0) ** (1.0 / self.gamma)
        # 2) 饱和度调整（向灰度中心缩放）
        gray = np.mean(img, axis=2, keepdims=True)
        img = gray + (img - gray) * self.saturation
        # 3) 色温偏移（暖 +R / -B）
        img[..., 0] = np.clip(img[..., 0] + self.temperature, 0.0, 1.0)
        img[..., 2] = np.clip(img[..., 2] - self.temperature, 0.0, 1.0)
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def attack(sample, quality: int = 85, seed: int = 42) -> dict:
    """颜色洗白攻击入口。"""
    image = sample["image"]
    out = JPEGReencodeAttack(quality=quality, seed=seed).apply(image)
    return {"image": out}
