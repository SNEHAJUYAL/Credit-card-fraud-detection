"""Score transactions with a saved model.

    python -m fraud_pipeline.predict ulb  transactions.csv     # columns: Time, V1..V28, Amount
    python -m fraud_pipeline.predict 2023 transactions.csv     # columns: V1..V28, Amount

Writes <input>_scored.csv with fraud_probability and a risk band. Band edges are illustrative defaults,
not calibrated production thresholds.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from .data import ROOT

MODELS = {
    "ulb": ("ulb_xgboost_best.json", "ulb_scaler.joblib", False),
    "2023": ("cc2023_xgboost.json", "cc2023_scaler.joblib", False),
}


def band(p: float) -> str:
    return "Low" if p < 0.25 else "Medium" if p < 0.5 else "High" if p < 0.75 else "Critical"


def score(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    model_file, scaler_file, _ = MODELS[kind]
    meta = joblib.load(ROOT / "models" / scaler_file)
    feats = meta["features"]
    model = XGBClassifier()
    model.load_model(ROOT / "models" / model_file)
    # The ULB model was trained on scaled inputs (resampling happened in scaled space); the 2023 model on raw.
    X = pd.DataFrame(meta["scaler"].transform(df[feats]), columns=feats) if kind == "ulb" else df[feats]
    out = df.copy()
    out["fraud_probability"] = model.predict_proba(X)[:, 1]
    out["risk_band"] = out["fraud_probability"].map(band)
    return out


if __name__ == "__main__":
    kind, path = sys.argv[1], Path(sys.argv[2])
    scored = score(kind, pd.read_csv(path))
    target = path.with_name(path.stem + "_scored.csv")
    scored.to_csv(target, index=False)
    print(scored["risk_band"].value_counts().to_string(), f"\nwrote {target}")
