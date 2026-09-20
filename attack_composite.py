#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击方法 6：隐藏组合攻击（Composite Hidden / 多步串联）

原理：
    单一攻击（JPEG/模糊/缩放/颜色/频域）各有侧重的取证痕迹去除方式，但也都可能
    被针对性防御检测出残留痕迹。组合攻击从"攻击池"中随机抽取 2~4 步串联执行，
    形成一条随机、不可预测的后处理链：

        jpeg -> resize_recover -> blur_sharpen -> color_laundering -> ...

    其核心思想是"分而治之 + 叠加隐匿"：
      - JPEG 主要破坏 8×8 分块 DCT 的高频能量分布；
      - 缩放-恢复抑制重采样/振铃伪影；
      - 模糊-锐化削弱 SRM 噪声残差与梯度统计；
      - 颜色洗白抹平颜色统计差异；
      - 频域洗白直接压制 FFT 高频带。
    多步串联使各种检测特征被逐层削弱，且随机链结构对防御方不可复现，鲁棒性更强。

    代价：多步叠加会累积一定失真，因此需控制步数与强度以通过保真度门槛（PSNR ≥ 24 dB）。

接口（平台契约）：
    attack(sample) -> {"image": RGB uint8 H×W×3，与输入同尺寸}
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

JPEG_QUALITIES = (55, 65, 75, 85)
DOWNSIZE_POOL = (192, 224)


def _to_pil(image):
    return Image.fromarray(np.asarray(image, dtype=np.uint8))


def _from_pil(img):
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _jpeg(image, quality):
    img = _to_pil(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return _from_pil(Image.open(buf))


def _resize_recover(image, down):
    img = _to_pil(image)
    w, h = img.size
    small = img.resize((down, down), Image.BICUBIC)
    return _from_pil(small.resize((w, h), Image.BICUBIC))


def _blur_sharpen(image):
    img = _to_pil(image).filter(ImageFilter.GaussianBlur(radius=1.2))
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=0))
    return _from_pil(img)


def _color_launder(image):
    img = np.asarray(image, dtype=np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0) ** (1.0 / 0.9)
    gray = np.mean(img, axis=2, keepdims=True)
    img = gray + (img - gray) * 1.1
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _freq_launder(image):
    img = np.asarray(image, dtype=np.float32)
    h, w, c = img.shape
    cy, cx = h // 2, w // 2
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt((yy / max(cy, 1)) ** 2 + (xx / max(cx, 1)) ** 2)
    mask = 1.0 / (1.0 + np.exp((r - 0.35) * 12.0))
    out = np.empty_like(img)
    for ch in range(c):
        f = np.fft.fftshift(np.fft.fft2(img[..., ch]))
        low = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask)))
        out[..., ch] = img[..., ch] * 0.4 + low * 0.6
    return np.clip(out, 0, 255).astype(np.uint8)


class JPEGReencodeAttack:
    """隐藏组合攻击（固定种子随机抽取 2~4 步攻击串联）。"""

    name = "composite_hidden"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        self.quality = int(quality)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self._pool = [
            ("jpeg", lambda s: _jpeg(None, int(s.choice(list(JPEG_QUALITIES))))),
            ("resize_recover", lambda s: _resize_recover(None, int(s.choice(DOWNSIZE_POOL)))),
            ("blur_sharpen", lambda s: _blur_sharpen(None)),
            ("color_laundering", lambda s: _color_launder(None)),
            ("frequency_laundering", lambda s: _freq_launder(None)),
        ]

    def apply(self, image: np.ndarray) -> np.ndarray:
        n = max(2, min(4, 3))  # 固定 3 步
        idxs = list(range(len(self._pool)))
        self._rng.shuffle(idxs)
        out = np.asarray(image, dtype=np.uint8)
        # 逐步串联：把上一步输出作为下一步输入
        for i in idxs[:n]:
            name, _factory = self._pool[i]
            if name == "jpeg":
                out = _jpeg(out, int(self._rng.choice(list(JPEG_QUALITIES))))
            elif name == "resize_recover":
                out = _resize_recover(out, int(self._rng.choice(DOWNSIZE_POOL)))
            elif name == "blur_sharpen":
                out = _blur_sharpen(out)
            elif name == "color_laundering":
                out = _color_launder(out)
            elif name == "frequency_laundering":
                out = _freq_launder(out)
        return out


def attack(sample, quality: int = 85, seed: int = 42) -> dict:
    """隐藏组合攻击入口。"""
    image = sample["image"]
    out = JPEGReencodeAttack(quality=quality, seed=seed).apply(image)
    return {"image": out}
