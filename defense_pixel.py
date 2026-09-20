#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防御方法 2：像素统计检测器（Pixel-Stat Detector）

原理：
    与频域检测器共享"特征 + 标准化 + 逻辑回归"范式，但特征完全来自空域/像素域，
    捕捉换脸 DeepFake 在颜色分布与边缘纹理上的统计差异。

    特征向量（15 维）：
      1. 每通道颜色统计（12 维）：RGB 三通道各取均值/标准差/极差(max-min)/
         95%-5% 分位差，刻画换脸区域的色差与动态范围异常。
      2. 灰度纹理统计（3 维）：灰度图水平梯度绝对均值、垂直梯度绝对均值、
         灰度标准差，刻画融合边界的锐利振铃与内部噪声导致的纹理变化。

    空域特征的优势：实现简单、计算快、对频域后处理（如 JPEG/低通）相对稳健，
    可作为频域检测器的互补，衡量攻击的"跨检测器迁移能力"。

接口（平台契约）：
    Defense(C=1.0, max_iter=500, seed=42).fit(images, labels).predict(image) -> float
"""

from __future__ import annotations

import numpy as np


def _extract_pixel_features(image: np.ndarray) -> np.ndarray:
    """15 维空域/像素统计特征向量。"""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    gray = np.mean(img, axis=2)

    feats = []
    # 1) 每通道颜色统计
    for c in range(3):
        ch = img[..., c]
        feats += [float(ch.mean()), float(ch.std()),
                  float(ch.max() - ch.min()),
                  float(np.percentile(ch, 95) - np.percentile(ch, 5))]
    # 2) 灰度纹理统计
    gx = np.diff(gray, axis=0)
    gy = np.diff(gray, axis=1)
    feats += [float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy))),
              float(np.std(gray))]
    return np.array(feats, dtype=np.float32)


class Defense:
    """像素统计检测器。"""

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
        X = np.stack([_extract_pixel_features(im) for im in images])
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
        feat = _extract_pixel_features(image).reshape(1, -1)
        feat = self._scaler.transform(feat)
        df = float(self._clf.decision_function(feat)[0])
        return float(1.0 / (1.0 + np.exp(-df)))
