"""Cross-dataset transfer: do V1-V28 mean the same thing in the ULB data and the 2023 data?

Train on one dataset's training split, score the ENTIRE other dataset (never seen in training).
Features: V1-V28 only (Amount and Time are on unrelated scales). Each dataset's V columns are standardised
with that dataset's own mean/std (unsupervised, no labels used). If the two datasets were unrelated,
transfer PR-AUC would sit at the fraud rate of the target dataset (0.0017 for ULB, 0.50 for 2023).

Run:  python -m fraud_pipeline.cross_dataset
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import ROOT, load_cc2023, load_ulb
from .evaluate import evaluate

SEED = 42
V = [f"V{i}" for i in range(1, 29)]


def _xgb(pos_weight):
    return XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist", eval_metric="aucpr",
                         n_jobs=-1, random_state=SEED, scale_pos_weight=pos_weight)


def run(log=print):
    ulb, new = load_ulb(), load_cc2023()
    sets = {"ULB": ulb, "2023": new}
    z = {k: pd.DataFrame(StandardScaler().fit_transform(d[V]), columns=V) for k, d in sets.items()}
    y = {k: d["Class"].astype(int).values for k, d in sets.items()}
    out = {"features": V, "rows": []}
    for src, dst in [("ULB", "2023"), ("2023", "ULB")]:
        Xtr, Xte, ytr, yte = train_test_split(z[src], y[src], test_size=0.2, stratify=y[src], random_state=SEED)
        model = _xgb(float((ytr == 0).sum() / (ytr == 1).sum())).fit(Xtr, ytr)
        in_domain = evaluate(yte, model.predict_proba(Xte)[:, 1])
        transfer = evaluate(y[dst], model.predict_proba(z[dst])[:, 1])
        out["rows"].append({"train_on": src, "scored_on": dst, "target_fraud_rate": float(y[dst].mean()),
                            "in_domain_pr_auc": in_domain["pr_auc"], "in_domain_recall": in_domain["recall"],
                            "transfer_pr_auc": transfer["pr_auc"], "transfer_roc_auc": transfer["roc_auc"],
                            "transfer_precision": transfer["precision"], "transfer_recall": transfer["recall"]})
        log(f"train {src} -> score all of {dst}: in-domain PR-AUC={in_domain['pr_auc']:.4f} | "
            f"transfer PR-AUC={transfer['pr_auc']:.4f} ROC-AUC={transfer['roc_auc']:.4f} "
            f"(chance PR-AUC = {y[dst].mean():.4f})")
    # same-meaning check: correlation of each V's fraud/legit mean gap across datasets (standardised units)
    gap = {k: (z[k][y[k] == 1].mean() - z[k][y[k] == 0].mean()) for k in sets}
    out["v_gap_correlation"] = float(np.corrcoef(gap["ULB"], gap["2023"])[0, 1])
    out["v_gap"] = {"ULB": gap["ULB"].round(3).to_dict(), "2023": gap["2023"].round(3).to_dict()}
    log(f"correlation of per-feature fraud-vs-legit mean gaps across the two datasets: {out['v_gap_correlation']:.3f}")
    (ROOT / "results" / "cross_dataset_results.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
