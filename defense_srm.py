#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防御方法 3：SRM 噪声残差检测器（SRM-only Detector）

原理：
    SRM（Spatial Rich Model，空间富模型）是数字图像取证领域的经典工具，其核心思想
    是用一族高通线性滤波器对图像卷积，得到"噪声残差"（noise residual），再统计残差
    的分布特征。真实自然图像的噪声残差遵循特定统计规律，而篡改/伪造（换脸融合）会
    破坏这种规律——融合边界与重采样会留下可测的高频残差异常。

    本检测器仅使用 SRM 残差统计（12 维），不依赖 DCT/FFT/颜色统计，是最"纯粹"的
    取证型检测器：
      对灰度图分别用 3 个标准 SRM 高通核卷积（5×5 二阶、5×5 一阶、3×3 拉普拉斯），
      每个残差图取 4 个统计量：均值 / 标准差 / 绝对均值 / 99%-1% 分位差。

    这 3×4=12 维特征被送入 StandardScaler + 逻辑回归，输出 fake_probability。
    相比频域检测器维度更低、更聚焦于噪声残差，对"只做颜色/亮度级攻击"的对抗样本
    具有互补的检出能力。

接口（平台契约）：
    Defense(C=1.0, max_iter=500, seed=42).fit(images, labels).predict(image) -> float
"""

from __future__ import annotations

import numpy as np

# SRM 高通噪声残差滤波器
_SRM_FILTERS = [
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
]


def _extract_srm_features(image: np.ndarray) -> np.ndarray:
    """12 维 SRM 噪声残差统计特征向量。"""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    gray = np.mean(img, axis=2)

    feats = []
    for kern in _SRM_FILTERS:
        res = ndi.convolve(gray, kern, mode="reflect")
        feats += [float(res.mean()), float(res.std()),
                  float(np.mean(np.abs(res))),
                  float(np.percentile(res, 99) - np.percentile(res, 1))]
    return np.array(feats, dtype=np.float32)


class Defense:
    """SRM 噪声残差检测器。"""

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
        X = np.stack([_extract_srm_features(im) for im in images])
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
        feat = _extract_srm_features(image).reshape(1, -1)
        feat = self._scaler.transform(feat)
        df = float(self._clf.decision_function(feat)[0])
        return float(1.0 / (1.0 + np.exp(-df)))
