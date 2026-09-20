#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防御方法对比评测：多域 MLP vs 频域/像素/SRM。

指标（与 test.py --stage defense 一致）：
    CleanAUC  = 干净测试集 ROC-AUC
    RobustAUC = 攻击池 A0 全部合法攻击后伪造图池化 + 真实图 的 ROC-AUC
    防御得分  = (CleanAUC + RobustAUC) / 2
"""
from __future__ import annotations

import importlib
import os
import sys
import time

import numpy as np
from sklearn import metrics as skm

_BASE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_BASE, "..", "Image-DeepFake", "code")
sys.path.insert(0, _CODE)
sys.path.insert(0, _BASE)

from dataset import load_benchmark  # noqa: E402
from attack import build_attack_pool  # noqa: E402


def _flat_region_ratio(image):
    img = np.asarray(image, dtype=np.int16)
    h, w, c = img.shape
    q = (img >> 4).astype(np.int16)
    code = q[..., 0] * 4096 + q[..., 1] * 64 + q[..., 2]
    gh, gw = 8, 8
    ys = np.linspace(0, h, gh, endpoint=False).astype(int)
    xs = np.linspace(0, w, gw, endpoint=False).astype(int)
    stds = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            y2 = ys[i + 1] if i + 1 < gh else h
            x2 = xs[j + 1] if j + 1 < gw else w
            block = code[y:y2, x:x2]
            if block.size:
                stds.append(float(block.std()))
    flat = sum(1 for s in stds if s < 1.0)
    return flat / max(len(stds), 1)


def check_legality(original, attacked):
    arr = np.asarray(attacked)
    ref = np.asarray(original)
    if arr.size == 0 or arr.ndim != 3 or arr.shape[2] != 3:
        return False
    if arr.shape[0] != ref.shape[0] or arr.shape[1] != ref.shape[1]:
        return False
    if not np.all(np.isfinite(arr.astype(np.float32))):
        return False
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    if _flat_region_ratio(arr8) > 0.5:
        return False
    return True


def evaluate(det, train, test):
    t0 = time.time()
    det.fit(train["images"], train["labels"])
    fit_t = time.time() - t0

    fake_mask = test["labels"] == 1
    real_mask = test["labels"] == 0
    fake_imgs = test["images"][fake_mask]
    real_imgs = test["images"][real_mask]
    n_fake = len(fake_imgs)

    clean_scores = np.array([det.predict(im) for im in test["images"]])
    clean_auc = skm.roc_auc_score(test["labels"], clean_scores)
    real_scores_clean = clean_scores[real_mask]

    pool = build_attack_pool()
    robust_scores, robust_labels = [], []
    per_attack_auc = {}
    for name, atk in pool:
        attacked = np.stack([atk.apply(im) for im in fake_imgs])
        valid = np.array([check_legality(fake_imgs[i], attacked[i])
                          for i in range(n_fake)])
        fake_scores = np.array([det.predict(im) for im in attacked])
        pa_scores = np.where(valid, fake_scores, 1.0)
        pa_lab = np.concatenate([np.ones(n_fake), np.zeros(len(real_imgs))])
        pa_sc = np.concatenate([pa_scores, real_scores_clean])
        per_attack_auc[name] = skm.roc_auc_score(pa_lab, pa_sc)
        robust_scores += list(fake_scores[valid])
        robust_labels += [1] * int(valid.sum())

    robust_scores += list(real_scores_clean)
    robust_labels += [0] * len(real_scores_clean)
    robust_auc = skm.roc_auc_score(np.asarray(robust_labels),
                                   np.asarray(robust_scores))
    defense_score = (clean_auc + robust_auc) / 2.0
    return clean_auc, robust_auc, defense_score, per_attack_auc, fit_t


def main():
    train, test = load_benchmark()
    mlp = importlib.import_module("defense_mlp")

    configs = [
        {"hidden_layer_sizes": (16,), "alpha": 0.1},
        {"hidden_layer_sizes": (32,), "alpha": 0.1},
        {"hidden_layer_sizes": (32, 16), "alpha": 0.01},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.01},
    ]

    print("=" * 76)
    print("防御方法对比评测（CleanAUC / RobustAUC / 防御得分）")
    print("=" * 76)

    results = []
    for cfg in configs:
        det = mlp.Defense(**cfg)
        ca, ra, ds, pa, ft = evaluate(det, train, test)
        results.append(("MLP" + str(cfg["hidden_layer_sizes"]), ca, ra, ds))
        print("\n[MLP] hidden={} alpha={}".format(cfg["hidden_layer_sizes"],
                                                  cfg["alpha"]))
        print("      CleanAUC={:.4f}  RobustAUC={:.4f}  防御得分={:.4f}  fit={:.1f}s"
              .format(ca, ra, ds, ft))
        for k in sorted(pa):
            print("        {:22s} AUC={:.4f}".format(k, pa[k]))

    for mod_name in ("defense_frequency", "defense_pixel", "defense_srm"):
        mod = importlib.import_module(mod_name)
        det = mod.Defense()
        ca, ra, ds, pa, ft = evaluate(det, train, test)
        results.append((mod_name, ca, ra, ds))
        print("\n[{}] CleanAUC={:.4f}  RobustAUC={:.4f}  防御得分={:.4f}  fit={:.1f}s"
              .format(mod_name, ca, ra, ds, ft))

    print("\n" + "=" * 76)
    print("汇总（按防御得分降序）")
    print("=" * 76)
    for name, ca, ra, ds in sorted(results, key=lambda x: -x[3]):
        print("  {:<24s} CleanAUC={:.4f}  RobustAUC={:.4f}  防御得分={:.4f}"
              .format(name, ca, ra, ds))


if __name__ == "__main__":
    main()
