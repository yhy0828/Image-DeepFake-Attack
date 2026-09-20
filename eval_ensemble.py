#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""集成防御实验：多域检测器取 max / 平均融合，对比单域检测器。"""
from __future__ import annotations

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
from eval_defense import check_legality  # noqa: E402

from defense_frequency import Defense as FreqDef  # noqa: E402
from defense_pixel import Defense as PixelDef  # noqa: E402
from defense_srm import Defense as SrmDef  # noqa: E402


class Ensemble:
    def __init__(self, mode="max"):
        self.mode = mode
        self._freq = FreqDef()
        self._pixel = PixelDef()
        self._srm = SrmDef()

    def fit(self, images, labels):
        self._freq.fit(images, labels)
        self._pixel.fit(images, labels)
        self._srm.fit(images, labels)
        return self

    def predict(self, image):
        p = [self._freq.predict(image), self._pixel.predict(image),
             self._srm.predict(image)]
        if self.mode == "max":
            return float(np.max(p))
        if self.mode == "mean":
            return float(np.mean(p))
        if self.mode == "median":
            return float(np.median(p))
        return float(np.max(p))


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

    print("=" * 76)
    print("集成防御实验（CleanAUC / RobustAUC / 防御得分）")
    print("=" * 76)

    for mode in ("max", "mean", "median"):
        det = Ensemble(mode=mode)
        ca, ra, ds, pa, ft = evaluate(det, train, test)
        print("\n[集成] mode={}".format(mode))
        print("      CleanAUC={:.4f}  RobustAUC={:.4f}  防御得分={:.4f}  fit={:.1f}s"
              .format(ca, ra, ds, ft))
        for k in sorted(pa):
            print("        {:22s} AUC={:.4f}".format(k, pa[k]))

    # 单域参考
    for name, cls in (("pixel", PixelDef), ("freq", FreqDef), ("srm", SrmDef)):
        det = cls()
        ca, ra, ds, pa, ft = evaluate(det, train, test)
        print("\n[{}] CleanAUC={:.4f}  RobustAUC={:.4f}  防御得分={:.4f}"
              .format(name, ca, ra, ds))


if __name__ == "__main__":
    main()
