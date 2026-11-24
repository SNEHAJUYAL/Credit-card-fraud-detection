"""Dataset loaders. Raw data lives in ./data (git-ignored: files exceed GitHub's 100 MB limit)."""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ULB "Credit Card Fraud Detection" (Kaggle mlg-ulb/creditcardfraud, the dataset used in video 2).
# Kaggle needs a login, so the identical dataset is fetched from its OpenML mirror and checksum-verified.
ULB_OPENML_URL = "https://api.openml.org/data/v1/download/1673544/creditcard.arff"
ULB_MD5 = "178bcf9bb1f31a3dfe12d0e577884add"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ulb() -> Path:
    DATA.mkdir(exist_ok=True)
    arff_path = DATA / "creditcard_ulb.arff"
    if not arff_path.exists():
        urllib.request.urlretrieve(ULB_OPENML_URL, arff_path)
    if _md5(arff_path) != ULB_MD5:
        raise RuntimeError(f"Checksum mismatch for {arff_path}; delete it and retry.")
    return arff_path


def load_ulb() -> pd.DataFrame:
    """ULB dataset: Time, V1-V28, Amount, Class (284,807 rows, 0.173% fraud)."""
    csv = DATA / "creditcard_ulb.csv"
    if csv.exists():
        return pd.read_csv(csv)
    from scipy.io import arff

    raw, _ = arff.loadarff(download_ulb())
    df = pd.DataFrame(raw)
    df["Class"] = df["Class"].map(lambda b: int(b.decode() if isinstance(b, bytes) else b))
    df.to_csv(csv, index=False)
    return df


def load_cc2023() -> pd.DataFrame:
    """Credit Card Fraud Detection Dataset 2023 (id, V1-V28, Amount, Class; 568,630 rows, 50/50)."""
    for p in (DATA / "creditcard_2023.csv", ROOT / "creditcard_2023.csv"):
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        "creditcard_2023.csv not found. Download it from "
        "https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023 "
        "and place it in ./data/ or the repo root."
    )


PAYSIM_COLUMNS = {"step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                  "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"}


def load_paysim(path=None, nrows=None) -> pd.DataFrame:
    """PaySim-style dataset from video 3 (Kaggle amanalisiddiqui/fraud-detection-dataset, ~6.36M rows).

    Kaggle requires a login for this file, so it must be downloaded manually into ./data/.
    """
    candidates = [Path(path)] if path else sorted(DATA.glob("*.csv"))
    for p in candidates:
        if not p.exists():
            continue
        head = pd.read_csv(p, nrows=2)
        if PAYSIM_COLUMNS.issubset(head.columns):
            return pd.read_csv(p, nrows=nrows)
    raise FileNotFoundError(
        "PaySim dataset not found in ./data/. Download 'Fraud Detection Dataset' from "
        "https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset (needs a free Kaggle login), "
        "unzip it, and put the CSV (e.g. 'AIML Dataset.csv') in ./data/."
    )
