#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地综合评测脚本：对 6 种攻击与 3 种防御做离线指标计算（供实验报告引用）。

用法：
    cd Image-DeepFake/code
    python ../../deepfake_solutions/local_test.py
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np

# 加入基准代码与解决方案目录
_BASE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_BASE, "..", "Image-DeepFake", "code")
sys.path.insert(0, _CODE)
sys.path.insert(0, _BASE)

from dataset import load_benchmark  # noqa: E402
from defense import FrequencyStatDefense, PixelStatDefense  # noqa: E402


def ssim(a, b, win_size=11, sigma=1.5, data_range=255.0):
    from scipy import ndimage as ndi
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ax = np.arange(win_size) - (win_size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    vals = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mu1 = ndi.convolve(x, k, mode="reflect")
        mu2 = ndi.convolve(y, k, mode="reflect")
        s1 = ndi.convolve(x * x, k, mode="reflect") - mu1 * mu1
        s2 = ndi.convolve(y * y, k, mode="reflect") - mu2 * mu2
        s12 = ndi.convolve(x * y, k, mode="reflect") - mu1 * mu2
        num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den = (mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2)
        vals.append(float(np.mean(num / np.maximum(den, 1e-12))))
    return float(np.mean(vals))


def psnr(a, b, data_range=255.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def evaluate_attacks():
    train, test = load_benchmark()
    fake = test["images"][test["labels"] == 1]
    dets = {"frequency_stat": FrequencyStatDefense(), "pixel_stat": PixelStatDefense()}
    for d in dets.values():
        d.fit(train["images"], train["labels"])

    attack_mods = ["attack_jpeg", "attack_resize", "attack_blur",
                   "attack_color", "attack_frequency", "attack_composite"]
    print("\n" + "=" * 72)
    print("攻击方法离线评测")
    print("=" * 72)
    for name in attack_mods:
        mod = importlib.import_module(name)
        attacked = np.stack([mod.attack({"image": im})["image"] for im in fake])
        asr = {}
        for dn, d in dets.items():
            scores = np.array([d.predict(im) for im in attacked])
            asr[dn] = float(np.mean(scores < 0.5))
        mean_asr = float(np.mean(list(asr.values())))
        m_ssim = float(np.mean([ssim(fake[i], attacked[i]) for i in range(len(fake))]))
        m_psnr = float(np.mean([psnr(fake[i], attacked[i]) for i in range(len(fake))]))
        print(f"{name:18s}  mean_ASR={mean_asr:.4f}  "
              f"ASR_freq={asr['frequency_stat']:.4f}  ASR_pixel={asr['pixel_stat']:.4f}  "
              f"SSIM={m_ssim:.4f}  PSNR={m_psnr:.2f}dB")


def evaluate_defenses():
    train, test = load_benchmark()
    def_mods = ["defense_frequency", "defense_pixel", "defense_srm"]
    print("\n" + "=" * 72)
    print("防御方法离线评测（干净测试集）")
    print("=" * 72)
    for name in def_mods:
        mod = importlib.import_module(name)
        det = mod.Defense()
        det.fit(train["images"], train["labels"])
        scores = np.array([det.predict(im) for im in test["images"]])
        labels = test["labels"]
        # 检出率 DR = 假样本中 pred>=0.5 的比例
        fake_scores = scores[labels == 1]
        real_scores = scores[labels == 0]
        dr = float(np.mean(fake_scores >= 0.5))
        fbr = float(np.mean(real_scores >= 0.5))
        acc = float(np.mean((scores >= 0.5).astype(int) == labels))
        # AUC
        try:
            from sklearn import metrics as skm
            auc = float(skm.roc_auc_score(labels, scores))
        except Exception:
            auc = float("nan")
        print(f"{name:18s}  检出率DR={dr:.4f}  FBR={fbr:.4f}  ACC={acc:.4f}  AUC={auc:.4f}")


if __name__ == "__main__":
    evaluate_attacks()
    evaluate_defenses()
