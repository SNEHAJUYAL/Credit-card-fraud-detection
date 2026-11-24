"""Train and save a compact model for the original 2023 dataset (balanced 50/50, 568,630 rows).

Run:  python -m fraud_pipeline.train_cc2023
Uses the same stratified 80/20 split (seed 42) as CreditCardDetection_Project.ipynb.
"""
import json
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import ROOT, load_cc2023
from .evaluate import evaluate

SEED = 42
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def run(log=print):
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    df = load_cc2023()
    X, y = df[FEATURES], df["Class"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(Xtr)
    rows = []

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(scaler.transform(Xtr), ytr)
    rows.append({"model": "LogisticRegression", "fit_seconds": round(time.time() - t0, 1),
                 **evaluate(yte.values, lr.predict_proba(scaler.transform(Xte))[:, 1], Xte["Amount"].values)})

    t0 = time.time()
    xgb = XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist", eval_metric="aucpr",
                        n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    rows.append({"model": "XGBoost", "fit_seconds": round(time.time() - t0, 1),
                 **evaluate(yte.values, xgb.predict_proba(Xte)[:, 1], Xte["Amount"].values)})
    for r in rows:
        log(f"[{r['model']}] PR-AUC={r['pr_auc']:.5f} P={r['precision']:.4f} R={r['recall']:.4f} FP={r['fp']} FN={r['fn']}")

    imp = dict(zip(FEATURES, xgb.feature_importances_.tolist()))
    out = {"dataset": "Credit Card Fraud Detection Dataset 2023", "n_rows": int(len(df)), "seed": SEED,
           "test_rows": int(len(yte)), "models": rows, "xgboost_importance": imp}
    (ROOT / "results" / "cc2023_results.json").write_text(json.dumps(out, indent=2))
    xgb.save_model(ROOT / "models" / "cc2023_xgboost.json")
    joblib.dump({"scaler": scaler, "features": FEATURES}, ROOT / "models" / "cc2023_scaler.joblib")
    log("saved results/cc2023_results.json and models/cc2023_xgboost.json")
    return out


if __name__ == "__main__":
    run()
