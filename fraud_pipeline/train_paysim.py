"""Fraud model for the PaySim-style dataset used in video 3 (Kaggle amanalisiddiqui/fraud-detection-dataset).

Run:  python -m fraud_pipeline.train_paysim [--nrows N] [--path FILE]
The ~186 MB CSV needs a Kaggle login, so download it manually into ./data/ first (see data.load_paysim).
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import ROOT, load_paysim
from .evaluate import evaluate

SEED = 42


def engineer(df: pd.DataFrame, log=print) -> tuple:
    """Return (X, y, amount). Drops account-name IDs (unique per row, no generalisable signal)."""
    df = df.copy()
    fraud_by_type = df.groupby("type")["isFraud"].agg(["sum", "count"])
    log("fraud by transaction type:\n" + fraud_by_type.to_string())
    risky = fraud_by_type.index[fraud_by_type["sum"] > 0]
    df = df[df["type"].isin(risky)]  # types with zero fraud add only easy negatives
    X = pd.DataFrame({
        "step": df["step"], "hour": df["step"] % 24, "amount": df["amount"],
        "oldbalanceOrg": df["oldbalanceOrg"], "newbalanceOrig": df["newbalanceOrig"],
        "oldbalanceDest": df["oldbalanceDest"], "newbalanceDest": df["newbalanceDest"],
        # balance bookkeeping mismatches: strong signals, but also partly an artefact of the simulator
        "errorBalanceOrig": df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"],
        "errorBalanceDest": df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"],
    })
    for t in sorted(risky):
        X[f"type_{t}"] = (df["type"] == t).astype(int)
    return X, df["isFraud"].astype(int), df["amount"].values


def run(path=None, nrows=None, rf_max_rows=1_000_000, log=print):
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    df = load_paysim(path, nrows)
    log(f"loaded {len(df):,} rows, fraud rate {100 * df['isFraud'].mean():.4f}%")
    X, y, amount = engineer(df, log)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    scaler = StandardScaler().fit(Xtr)
    rows = []

    def record(name, model, Xa, Xb, t0, extra=None):
        proba = model.predict_proba(Xb)[:, 1]
        r = {"model": name, "fit_seconds": round(time.time() - t0, 1), **(extra or {}),
             **evaluate(yte.values, proba, amount[te])}
        rows.append(r)
        log(f"[{name}] PR-AUC={r['pr_auc']:.4f} P={r['precision']:.3f} R={r['recall']:.3f} FP={r['fp']} FN={r['fn']}")

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(scaler.transform(Xtr), ytr)
    record("LogisticRegression(balanced)", lr, None, scaler.transform(Xte), t0)

    t0 = time.time()
    xgb = XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist", eval_metric="aucpr", n_jobs=-1,
                        scale_pos_weight=pos_weight, random_state=SEED).fit(Xtr, ytr)
    record("XGBoost(scale_pos_weight)", xgb, None, Xte, t0)

    # Random Forest: keep every fraud row, thin out the legitimate rows to stay tractable
    t0 = time.time()
    pos = ytr[ytr == 1].index
    neg = ytr[ytr == 0]
    neg = neg.sample(n=min(len(neg), max(rf_max_rows - len(pos), 1)), random_state=SEED).index
    sel = pos.union(neg)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced_subsample",
                                min_samples_leaf=2, random_state=SEED).fit(Xtr.loc[sel], ytr.loc[sel])
    record("RandomForest(thinned negatives)", rf, None, Xte, t0, {"train_rows": int(len(sel))})

    out = {"dataset": "PaySim (Kaggle amanalisiddiqui/fraud-detection-dataset)", "n_rows": int(len(df)),
           "test_rows": int(len(yte)), "test_fraud": int(yte.sum()), "models": rows,
           "xgboost_importance": dict(zip(X.columns, xgb.feature_importances_.tolist()))}
    (ROOT / "results" / "paysim_results.json").write_text(json.dumps(out, indent=2))
    xgb.save_model(ROOT / "models" / "paysim_xgboost.json")
    joblib.dump({"scaler": scaler, "features": list(X.columns)}, ROOT / "models" / "paysim_scaler.joblib")
    log("saved results/paysim_results.json and models/paysim_xgboost.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path")
    ap.add_argument("--nrows", type=int)
    a = ap.parse_args()
    run(a.path, a.nrows)
