#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击方法 2：缩放-恢复（Resize & Recover / 重采样）

原理：
    DeepFake 的融合边界痕迹带有高频振铃与重采样伪影，这些精细结构对"下采样再
    上采样"十分敏感。把图像先缩小到较小的中间尺寸（如 192/224），再用双三次插值
    放大回原尺寸（256），相当于对图像做一次低通平滑重采样：

      - 下采样（平均/插值）抑制了高频振铃与局部纹理噪声；
      - 上采样插值抹平了原融合边界的锐利边缘，使残留的高频取证痕迹进一步弱化。

    相比 JPEG，缩放-恢复不引入分块伪影，视觉上更接近"轻微模糊"，但同样能有效
    降低依赖高频特征的检测器置信度。插值核（双线性/双三次/兰佐斯）决定平滑强度。

接口（平台契约）：
    attack(sample) -> {"image": RGB uint8 H×W×3，与输入同尺寸}
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DOWNSIZE_POOL = (192, 224)


class JPEGReencodeAttack:
    """缩放-恢复攻击（下采样到 192/224 再双三次放大回原尺寸）。"""

    name = "resize_recover"

    def __init__(self, quality: int = 85, seed: int = 42) -> None:
        self.down_size = 192 if int(quality) <= 70 else 224
        self.seed = int(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(np.asarray(image, dtype=np.uint8))
        w, h = img.size
        small = img.resize((self.down_size, self.down_size), Image.BICUBIC)
        recover = small.resize((w, h), Image.BICUBIC)
        return np.asarray(recover.convert("RGB"), dtype=np.uint8)


def attack(sample, quality: int = 85, seed: int = 42) -> dict:
    """缩放-恢复攻击入口。"""
    image = sample["image"]
    out = JPEGReencodeAttack(quality=quality, seed=seed).apply(image)
    return {"image": out}
