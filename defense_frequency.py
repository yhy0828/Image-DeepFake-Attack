#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防御方法 1：频域统计检测器（Frequency-Stat Detector）

原理：
    检测器基于"特征提取 + 标准化 + 逻辑回归"的经典机器学习范式，输出 fake_probability。

    特征向量（31 维）由四部分构成，专为捕捉 DeepFake 的频域取证痕迹而设计：
      1. SRM 高通噪声残差统计（12 维）：使用 3 个标准 SRM（Spatial Rich Model）
         高通滤波器对灰度图卷积，提取噪声/边缘残差，各取均值/标准差/绝对均值/
         99%-1% 分位差。伪造图的融合边界振铃会让这些高通残差响应异常偏高。
      2. 8×8 分块 DCT 能量谱（6 维）：对灰度图做 8×8 分块 2D-DCT，累加各块幅度谱，
         取 5 个代表性系数（低频 0/1/9 与高频 56/63）的 log1p 能量，以及高频/低频
         能量比。JPEG 编码与伪造痕迹都会改变块级 DCT 的高频能量分布。
      3. 全局 FFT 幅度谱频带统计（4 维）：对整图做 2D FFT，取低频带（半径<15%）
         与高频带（半径>40%）的 log1p 幅度均值与标准差。
      4. 每通道颜色统计（9 维）：RGB 三通道的均值/标准差/极差，捕捉换脸色差。

    训练：在（真实+伪造）训练集上提取特征 -> StandardScaler 标准化 -> 逻辑回归拟合。
    推理：特征 -> 标准化 -> 逻辑回归决策值 -> sigmoid -> P(fake|image) ∈ [0,1]。

接口（平台契约）：
    Defense(C=1.0, max_iter=500, seed=42).fit(images, labels).predict(image) -> float
"""

from __future__ import annotations

import numpy as np

# SRM 高通噪声残差滤波器（标准 17 个中的 3 个）
_SRM_FILTERS = [
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
]


def _dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II（基于 scipy.fft.dct，逐轴正交归一）。"""
    from scipy.fft import dct
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _extract_frequency_features(image: np.ndarray) -> np.ndarray:
    """31 维频域 + 噪声残差特征向量。"""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    gray = np.mean(img, axis=2)

    feats = []
    # 1) SRM 噪声残差统计
    for kern in _SRM_FILTERS:
        res = ndi.convolve(gray, kern, mode="reflect")
        feats += [float(res.mean()), float(res.std()),
                  float(np.mean(np.abs(res))),
                  float(np.percentile(res, 99) - np.percentile(res, 1))]
    # 2) 8×8 分块 DCT 能量
    h, w = gray.shape
    bh, bw = 8, 8
    acc = np.zeros((bh, bw), dtype=np.float32)
    count = 0
    for i in range(0, h - bh + 1, bh):
        for j in range(0, w - bw + 1, bw):
            acc += np.abs(_dct2(gray[i:i + bh, j:j + bw]))
            count += 1
    if count > 0:
        acc /= count
    flat = acc.flatten()
    feats += [float(np.log1p(flat[k])) for k in (0, 1, 9, 56, 63)]
    low = float(np.sum(acc[:2, :2]))
    high = float(np.sum(acc[4:, 4:]))
    feats += [float(np.log1p(high / (low + 1e-6)))]
    # 3) 全局 FFT 幅度谱频带统计
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = r < min(h, w) * 0.15
    high_mask = r > min(h, w) * 0.4
    feats += [float(mag[low_mask].mean()), float(mag[high_mask].mean()),
              float(mag[low_mask].std()), float(mag[high_mask].std())]
    # 4) 每通道颜色统计
    for c in range(3):
        ch = img[..., c]
        feats += [float(ch.mean()), float(ch.std()), float(ch.max() - ch.min())]
    return np.array(feats, dtype=np.float32)


class Defense:
    """频域统计检测器（交付 Baseline）。"""

    name = "defense"

    def __init__(self, C: float = 1.0, max_iter: int = 500, seed: int = 42, **kwargs):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        self._fitted = False

    def fit(self, images, labels):
        labels = np.asarray(labels, dtype=np.int64)
        X = np.stack([_extract_frequency_features(im) for im in images])
        if len(np.unique(labels)) < 2:
            self._fitted = False
            return self
        Xs = self._scaler.fit_transform(X)
        self._clf.fit(Xs, labels)
        self._fitted = True
        return self

    def predict(self, image) -> float:
        if not self._fitted:
            return 0.5
        feat = _extract_frequency_features(image).reshape(1, -1)
        feat = self._scaler.transform(feat)
        df = float(self._clf.decision_function(feat)[0])
        return float(1.0 / (1.0 + np.exp(-df)))
