"""Stratified 5-fold cross-validation of the main ULB contenders.

A single 80/20 split holds only 98 fraud cases, so small PR-AUC gaps are noise. Cross-validation pools
all 492 frauds across 5 test folds and reports mean +/- std. Resampling is applied inside each training fold only.

Run:  python -m fraud_pipeline.cv_ulb
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data import ROOT, load_ulb
from .evaluate import evaluate
from .train_ulb import FEATURES, SEED, _make_model, _resamplers

COMBOS = [("LogReg", "undersample (video 2)"), ("RandomForest", "none"), ("RandomForest", "SMOTE"),
          ("XGBoost", "none"), ("XGBoost", "class_weight"), ("XGBoost", "SMOTE")]


def run(n_splits=5, log=print):
    df = load_ulb()
    X, y = df[FEATURES], df["Class"].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = {f"{m}|{s}": [] for m, s in COMBOS}
    for k, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr = pd.DataFrame(scaler.transform(X.iloc[tr]), columns=FEATURES)
        Xte = pd.DataFrame(scaler.transform(X.iloc[te]), columns=FEATURES)
        ytr, yte, amt = y.iloc[tr].reset_index(drop=True), y.iloc[te].values, X.iloc[te]["Amount"].values
        pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
        for m, s in COMBOS:
            t0 = time.time()
            if m == "LogReg":
                fraud = ytr[ytr == 1].index
                keep = fraud.union(ytr[ytr == 0].sample(n=len(fraud), random_state=SEED + k).index)
                model = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr.loc[keep], ytr.loc[keep])
            else:
                sampler = _resamplers()[s]
                if sampler is None or sampler == "class_weight":
                    Xf, yf = Xtr, ytr
                else:
                    Xf, yf = sampler.fit_resample(Xtr, ytr)
                    Xf = pd.DataFrame(Xf, columns=FEATURES)
                model = _make_model(m, s, pos_weight).fit(Xf, yf)
            r = evaluate(yte, model.predict_proba(Xte)[:, 1], amt)
            folds[f"{m}|{s}"].append(r)
            log(f"fold {k + 1}/{n_splits} [{m}|{s}] PR-AUC={r['pr_auc']:.4f} P={r['precision']:.3f} R={r['recall']:.3f} ({time.time() - t0:.0f}s)")
    summary = {}
    for key, rs in folds.items():
        d = {}
        for metric in ("pr_auc", "roc_auc", "precision", "recall", "f1", "fp", "fn"):
            v = np.array([r[metric] for r in rs], dtype=float)
            d[metric] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "folds": v.tolist()}
        summary[key] = d
    out = {"n_splits": n_splits, "seed": SEED, "n_rows": int(len(df)), "n_fraud": int(y.sum()), "summary": summary}
    (ROOT / "results" / "ulb_cv_results.json").write_text(json.dumps(out, indent=2))
    log("saved results/ulb_cv_results.json")
    return out


if __name__ == "__main__":
    run()
