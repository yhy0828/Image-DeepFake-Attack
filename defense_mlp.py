#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防御方法 4：多域特征融合 + MLP 神经网络检测器（Multi-Domain MLP Detector）

原理：
    单一特征域的检测器容易被针对性后处理攻击（频域洗白、缩放、颜色洗白）逐个击破：
      - 频域统计（DCT/FFT/SRM）被"频域洗白"直接压制高频而失效；
      - 空域/颜色统计被"颜色洗白"抹平；
      - 纯 SRM 噪声残差被"模糊-锐化"削弱。
    本方法把多个互补特征域拼成一个融合特征向量，再送入多层感知机（MLP）学习非线性
    判别边界，让攻击者难以同时抹干净所有维度，从而提高对未知后处理的鲁棒性。

    融合特征（37 维）：
      1. SRM 高通噪声残差统计（12 维）：3 个 SRM 核卷积灰度图，各取均值/标准差/
         绝对均值/99%-1% 分位差，刻画融合边界的振铃与高频噪声残留；
      2. 8×8 分块 DCT 能量谱（6 维）：5 个代表性系数（低频 0/1/9 与高频 56/63）的
         log1p 能量 + 高低频能量比，刻画 JPEG/伪造痕迹的块级频域分布；
      3. 全局 FFT 幅度谱频带统计（4 维）：低频带（<15%）与高频带（>40%）的 log1p
         幅度均值与标准差；
      4. 每通道颜色统计（12 维）：RGB 三通道均值/标准差/极差/95%-5% 分位差；
      5. 空域梯度统计（3 维）：水平/垂直梯度绝对均值与灰度标准差。

    分类器：StandardScaler 标准化 + MLPClassifier（sklearn 多层感知机，即轻量全连接
    神经网络）。相比逻辑回归，MLP 能学习特征间非线性交互，在跨域特征上有更强拟合
    能力；用 L2 正则化（alpha）抑制小样本过拟合。

接口（平台契约）：
    Defense(hidden_layer_sizes=(64,32), alpha=0.01, max_iter=2000, seed=42)
        .fit(images, labels).predict(image) -> float ∈ [0,1]
"""

import numpy as np

# SRM 高通噪声残差滤波器（标准 17 个中的 3 个）
_SRM_FILTERS = [
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
]


def _dct2(block):
    """2D DCT-II（基于 scipy.fft.dct，逐轴正交归一）。"""
    from scipy.fft import dct
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def _extract_fusion_features(image):
    """37 维多域融合特征向量。"""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    gray = np.mean(img, axis=2)
    feats = []

    # 1) SRM 高通噪声残差统计（12 维）
    for kern in _SRM_FILTERS:
        res = ndi.convolve(gray, kern, mode="reflect")
        feats += [float(res.mean()), float(res.std()),
                  float(np.mean(np.abs(res))),
                  float(np.percentile(res, 99) - np.percentile(res, 1))]

    # 2) 8×8 分块 DCT 能量（6 维）
    h, w = gray.shape
    acc = np.zeros((8, 8), dtype=np.float32)
    count = 0
    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            acc += np.abs(_dct2(gray[i:i + 8, j:j + 8]))
            count += 1
    if count > 0:
        acc /= count
    flat = acc.flatten()
    feats += [float(np.log1p(flat[k])) for k in (0, 1, 9, 56, 63)]
    low = float(np.sum(acc[:2, :2]))
    high = float(np.sum(acc[4:, 4:]))
    feats += [float(np.log1p(high / (low + 1e-6)))]

    # 3) 全局 FFT 幅度谱频带统计（4 维）
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = r < min(h, w) * 0.15
    high_mask = r > min(h, w) * 0.4
    feats += [float(mag[low_mask].mean()), float(mag[high_mask].mean()),
              float(mag[low_mask].std()), float(mag[high_mask].std())]

    # 4) 每通道颜色统计（12 维：均值/标准差/极差/95%-5% 分位差）
    for c in range(3):
        ch = img[..., c]
        feats += [float(ch.mean()), float(ch.std()),
                  float(ch.max() - ch.min()),
                  float(np.percentile(ch, 95) - np.percentile(ch, 5))]

    # 5) 空域梯度统计（3 维）
    gx = np.diff(gray, axis=0)
    gy = np.diff(gray, axis=1)
    feats += [float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy))),
              float(np.std(gray))]

    return np.nan_to_num(np.array(feats, dtype=np.float32))


class Defense:
    """多域特征融合 + MLP 神经网络检测器。"""

    name = "defense"

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=0.01,
                 max_iter=2000, seed=42, **kwargs):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed
        self._scaler = StandardScaler()
        self._clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                  alpha=alpha, max_iter=max_iter,
                                  random_state=seed)
        self._fitted = False

    def fit(self, images, labels):
        labels = np.asarray(labels, dtype=np.int64)
        X = np.stack([_extract_fusion_features(im) for im in images])
        if len(np.unique(labels)) < 2:
            self._fitted = False
            return self
        Xs = self._scaler.fit_transform(X)
        self._clf.fit(Xs, labels)
        self._fitted = True
        return self

    def predict(self, image):
        if not self._fitted:
            return 0.5
        feat = _extract_fusion_features(image).reshape(1, -1)
        feat = self._scaler.transform(feat)
        proba = self._clf.predict_proba(feat)
        return float(proba[0, 1])
