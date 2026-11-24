"""Smoke test: runs the PaySim pipeline on a tiny SYNTHETIC file with the same schema.

This only proves the code executes end to end. The synthetic data says nothing about real fraud behaviour,
and its outputs are written to a temp directory, never to ./results.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_pipeline import train_paysim


def make_fake(n=40_000, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    t = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n, p=[.35, .1, .3, .22, .03])
    amount = rng.lognormal(7, 1.5, n).round(2)
    old = rng.lognormal(8, 2, n).round(2)
    fraud = ((np.isin(t, ["TRANSFER", "CASH_OUT"])) & (rng.rand(n) < 0.04)).astype(int)
    amount = np.where(fraud == 1, old, amount)  # fake "drain the account" pattern
    new = np.where(fraud == 1, 0.0, np.maximum(old - amount, 0)).round(2)
    return pd.DataFrame({
        "step": rng.randint(1, 744, n), "type": t, "amount": amount,
        "nameOrig": [f"C{i}" for i in range(n)], "oldbalanceOrg": old, "newbalanceOrig": new,
        "nameDest": [f"M{i}" for i in range(n)], "oldbalanceDest": rng.lognormal(8, 2, n).round(2),
        "newbalanceDest": rng.lognormal(8, 2, n).round(2), "isFraud": fraud, "isFlaggedFraud": 0,
    })


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "fake_paysim.csv"
        make_fake().to_csv(csv, index=False)
        train_paysim.ROOT = Path(d)  # keep outputs out of the real results folder
        out = train_paysim.run(path=str(csv), log=lambda *a: None)
        assert {m["model"] for m in out["models"]} >= {"XGBoost(scale_pos_weight)", "LogisticRegression(balanced)"}
        assert (Path(d) / "results" / "paysim_results.json").exists()
        print("PaySim smoke test passed on synthetic data (", len(out["models"]), "models trained )")
