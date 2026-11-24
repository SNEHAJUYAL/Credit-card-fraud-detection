"""ULB credit card fraud experiments replicating the approaches from videos 1 and 2.

Video 2 (Siddhardhan): under-sample legitimate transactions to a 50/50 set, fit Logistic Regression.
Video 1 (JP Infotech): Random Forest / XGBoost with synthetic over-sampling (SMOTE family).

Run:  python -m fraud_pipeline.train_ulb
Every resampler is applied to the TRAINING split only. All models are scored on the same untouched,
naturally imbalanced test split, so numbers are comparable and free of resampling leakage.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import ROOT, load_ulb
from .evaluate import evaluate

SEED = 42
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


def prepare(df: pd.DataFrame, test_size=0.2, seed=SEED) -> dict:
    X, y = df[FEATURES], df["Class"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    amount_te = Xte["Amount"].values.copy()
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), columns=FEATURES, index=Xtr.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=FEATURES, index=Xte.index)
    return {"Xtr": Xtr_s, "Xte": Xte_s, "ytr": ytr, "yte": yte, "amount_te": amount_te, "scaler": scaler}


def video2_logreg(S: dict, seed=SEED) -> dict:
    """Under-sample legitimate rows to match fraud, fit Logistic Regression (as in video 2)."""
    Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
    fraud_idx = ytr[ytr == 1].index
    legit_idx = ytr[ytr == 0].sample(n=len(fraud_idx), random_state=seed).index
    idx = fraud_idx.union(legit_idx)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr.loc[idx], ytr.loc[idx])

    te_fraud = yte[yte == 1].index
    te_legit = yte[yte == 0].sample(n=len(te_fraud), random_state=seed).index
    bal = te_fraud.union(te_legit)
    proba = model.predict_proba(Xte)[:, 1]
    return {
        "train_rows": int(len(idx)),
        "train_accuracy": float(accuracy_score(ytr.loc[idx], model.predict(Xtr.loc[idx]))),
        "balanced_test_rows": int(len(bal)),
        "balanced_test_accuracy": float(accuracy_score(yte.loc[bal], model.predict(Xte.loc[bal]))),
        "full_test": evaluate(yte.values, proba, S["amount_te"]),
        "model": model, "proba": proba,
    }


def _resamplers(seed=SEED) -> dict:
    return {
        "none": None,
        "class_weight": "class_weight",
        "SMOTE": SMOTE(random_state=seed),
        "ADASYN": ADASYN(random_state=seed),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=seed),
        "KMeansSMOTE": KMeansSMOTE(
            random_state=seed, cluster_balance_threshold=0.02,
            kmeans_estimator=MiniBatchKMeans(n_clusters=20, n_init=3, random_state=seed)),
    }


def _make_model(name: str, strategy: str, pos_weight: float, seed=SEED):
    cw = strategy == "class_weight"
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed,
                                      class_weight="balanced_subsample" if cw else None)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                             eval_metric="aucpr", n_jobs=-1, random_state=seed,
                             scale_pos_weight=pos_weight if cw else 1.0)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=50, random_state=seed)
    raise ValueError(name)


def video1_resampling(S: dict, log=print, only=None) -> tuple:
    """RF / XGBoost (and an MLP) under each imbalance strategy. Returns (rows, probas, importances, xgb_models)."""
    Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    plan = [(m, s) for s in _resamplers() for m in ("RandomForest", "XGBoost")]
    plan += [("MLP", "none"), ("MLP", "SMOTE")]
    rows, probas, importances, xgb_models = [], {}, {}, {}
    cache = {}
    for model_name, strategy in plan:
        if only and (model_name, strategy) not in only:
            continue
        key = f"{model_name}|{strategy}"
        t0 = time.time()
        try:
            sampler = _resamplers()[strategy]
            if strategy not in cache:
                if sampler is None or sampler == "class_weight":
                    cache[strategy] = (Xtr, ytr, 0.0)
                else:
                    ts = time.time()
                    Xr, yr = sampler.fit_resample(Xtr, ytr)
                    cache[strategy] = (pd.DataFrame(Xr, columns=FEATURES), pd.Series(yr), time.time() - ts)
            Xfit, yfit, resample_s = cache[strategy]
            model = _make_model(model_name, strategy, pos_weight).fit(Xfit, yfit)
            proba = model.predict_proba(Xte)[:, 1]
            m = evaluate(yte.values, proba, S["amount_te"])
            row = {"experiment": "video1", "model": model_name, "strategy": strategy, "status": "ok",
                   "train_rows": int(len(yfit)), "train_fraud_rows": int(yfit.sum()),
                   "resample_seconds": round(resample_s, 1), "fit_seconds": round(time.time() - t0 - 0, 1), **m}
            probas[key] = proba.astype("float64")
            if hasattr(model, "feature_importances_"):
                importances[key] = model.feature_importances_.tolist()
            if model_name == "XGBoost":
                xgb_models[strategy] = model
            log(f"[{key}] PR-AUC={m['pr_auc']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} FP={m['fp']} FN={m['fn']} ({row['fit_seconds']}s)")
        except Exception as e:  # keep going; failures are reported, not hidden
            row = {"experiment": "video1", "model": model_name, "strategy": strategy,
                   "status": f"failed: {type(e).__name__}: {str(e)[:200]}"}
            log(f"[{key}] FAILED -> {row['status']}")
        rows.append(row)
    return rows, probas, importances, xgb_models


def run(log=print, only=None):
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = load_ulb()
    S = prepare(df)
    log(f"train {len(S['ytr']):,} rows ({int(S['ytr'].sum())} fraud) | test {len(S['yte']):,} rows ({int(S['yte'].sum())} fraud)")

    v2 = video2_logreg(S)
    log(f"[video2 LogReg] balanced-test accuracy={v2['balanced_test_accuracy']:.4f} | "
        f"full-test PR-AUC={v2['full_test']['pr_auc']:.4f} precision={v2['full_test']['precision']:.4f} "
        f"recall={v2['full_test']['recall']:.4f} FP={v2['full_test']['fp']}")
    rows, probas, importances, xgb_models = video1_resampling(S, log=log, only=only)

    probas["LogReg|undersample (video 2)"] = v2["proba"].astype("float64")
    ok = [r for r in rows if r["status"] == "ok"]
    best = max(ok, key=lambda r: r["pr_auc"]) if ok else None
    best_xgb = max((r for r in ok if r["model"] == "XGBoost"), key=lambda r: r["pr_auc"], default=None)

    out = {
        "dataset": "ULB creditcardfraud (OpenML mirror of Kaggle mlg-ulb/creditcardfraud)",
        "n_rows": int(len(df)), "n_fraud": int(df["Class"].sum()),
        "test_rows": int(len(S["yte"])), "test_fraud": int(S["yte"].sum()),
        "features": FEATURES, "seed": SEED,
        "video2": {k: v for k, v in v2.items() if k not in ("model", "proba")},
        "video1": rows, "importances": importances,
        "best_overall": {k: best[k] for k in ("model", "strategy", "pr_auc")} if best else None,
        "best_xgboost": {k: best_xgb[k] for k in ("model", "strategy", "pr_auc")} if best_xgb else None,
    }
    (RESULTS / "ulb_results.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(RESULTS / "ulb_test_probas.npz", y_test=S["yte"].values.astype("int8"),
                        amount=S["amount_te"].astype("float64"), **{k.replace("|", "__"): v for k, v in probas.items()})

    joblib.dump({"scaler": S["scaler"], "features": FEATURES}, MODELS / "ulb_scaler.joblib")
    joblib.dump(v2["model"], MODELS / "ulb_logreg_undersample.joblib")
    if best_xgb:
        xgb_models[best_xgb["strategy"]].save_model(MODELS / "ulb_xgboost_best.json")
    log("saved results/ulb_results.json, results/ulb_test_probas.npz and compact models in models/")
    return out


if __name__ == "__main__":
    run()
