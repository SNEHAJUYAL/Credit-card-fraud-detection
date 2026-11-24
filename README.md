# Credit Card Fraud Risk Analytics & Intelligent Detection System
### Original baseline analysis: Jonathan Wan — extended into a full business analytics & ML project

This project uses the `Credit Card Fraud Detection Dataset 2023`, which contains **568,630** anonymized, PCA-transformed credit-card transactions. `CreditCardDetection_Project.ipynb` treats the dataset as the foundation for an end-to-end fraud-risk analytics solution: data quality assessment, exploratory and fraud-pattern analysis, an amount-segmentation study, a five-model machine-learning comparison, SHAP explainability, a configurable cost-of-fraud framework, risk-based scoring, and an operational fraud workflow with business recommendations — all computed directly from the dataset, not assumed.

## Key Features:
- `id` : Unique identifier for each transaction
- `V1-V28` : Anonymized/PCA-transformed transaction features. **No real-world meaning (e.g. location, device, time) is disclosed by the data source for these variables**, so this project reports them purely as statistical predictors rather than inventing business interpretations for them.
- `Amount` : The transaction amount
- `Class` : Binary label indicating whether the transaction is fraudulent (1) or not (0)

*Data Source: The dataset was collected from credit card transactions made by European cardholders in 2023, with sensitive information removed to ensure privacy and compliance with ethical guidelines.*
https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

## Goal:
Build a professional, business-framed fraud-risk analytics solution — not just an EDA notebook — that quantifies fraud exposure, characterizes what separates fraud from legitimate transactions, compares multiple ML models on business-relevant metrics, and translates model output into a risk-based operating workflow with concrete, data-traceable recommendations.

## Important dataset characteristics (verified from the data, not assumed):
- **The dataset is exactly class-balanced** (50.00% fraud / 50.00% legitimate) — the opposite of a real-world card portfolio, where fraud is typically a small fraction of one percent of transactions. This is treated throughout as a benchmark-construction property, not a real-world fraud rate.
- **`Amount` is close to uniformly distributed** ($50–$24,040, skew ≈ 0) and is essentially uncorrelated with `Class` — fraud rate is nearly flat (~50%) across every amount segment. Amount-based rules would not meaningfully separate fraud from legitimate transactions in this dataset.
- No missing values and no duplicate rows were found.

## Results (from the current model run in the notebook):
- Five models were compared on Precision, Recall, F1, ROC-AUC and PR-AUC (not accuracy alone): Logistic Regression, Decision Tree, Bagging, Random Forest, and XGBoost.
- **Random Forest** was selected as the best model — PR-AUC 1.0000, 100.00% recall, 99.97% precision on a held-out, stratified test set (0 false negatives, 19 false positives out of 113,726 test transactions).
- All ensemble models scored above 99.9% accuracy; this is called out in the notebook as a sign of unusually clean class separability in this specific benchmark (consistent with its exact balance and uniform `Amount` distribution), not an expectation of real-world production performance.
- **Revised feature-importance finding:** across Random Forest, XGBoost, and SHAP, `V14` and `V10` are consistently top-ranked anonymized predictors, confirming two-thirds of the original `V1, V10, V14` claim. `V1`, however, does **not** rank in the top 5 by Random Forest or SHAP importance — it only appears via XGBoost — so the original top-3 claim is only partially supported once nonlinear, trained-model importance is used instead of simple correlation. The strongest predictors identified in this run are `V14`, `V4`, `V12`, and `V10`.

See `CreditCardDetection_Project.ipynb` for the full analysis, business interpretation of every chart, the cost-of-fraud and risk-scoring frameworks, and prioritized business recommendations.

## Documentation
[`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) explains the whole project from basics to advanced: the STAR-format project story, why each model, library and line of code was used, how to reproduce it, a decision log, limitations, and interview Q&A.

## Run it yourself
Download `creditcard_2023.csv` from the Kaggle link above and place it next to the notebook (it is git-ignored because it exceeds GitHub's 100 MB file limit), then `pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter` and open the notebook. Full execution takes roughly 15-20 minutes.

## Part B: three video approaches, three datasets, trained models
`Multi_Dataset_Fraud_Detection.ipynb` (executed) extends the project with the approaches and datasets from three YouTube tutorials, evaluated under one protocol (stratified split, resampling on the training split only, PR-AUC as the headline metric, 5-fold cross-validation).

| Source | Approach | Dataset | Status |
|---|---|---|---|
| Video 1, JP Infotech | Random Forest / XGBoost + SMOTE, ADASYN, Borderline-SMOTE, KMeans-SMOTE | Not named in the description (base paper uses proprietary data) | Methods run on ULB. AGSS sampler, LSTM and Flask app not replicated |
| Video 2, Siddhardhan | Logistic Regression | ULB `mlg-ulb/creditcardfraud` | Replicated (OpenML mirror, MD5-verified) |
| Video 3, Onur Baltaci | Fraud detection project | PaySim-style, Kaggle `amanalisiddiqui/fraud-detection-dataset` | Pipeline written and smoke-tested on synthetic data. **Not run on real data**: needs a Kaggle login to download |
| Original | RF / XGBoost / Logistic Regression | 2023 dataset | Existing notebook, plus saved models |

Key results on the real, imbalanced ULB data (492 frauds in 284,807 rows):
- Logistic Regression on an under-sampled 50/50 set scores 94.4% accuracy on a balanced hold-out, but on the real distribution only 3.8% of its alerts are fraud (2,258 false alarms for 90 frauds found). PR-AUC 0.66.
- Random Forest and XGBoost reach cross-validated PR-AUC of about 0.85 (± 0.03). SMOTE-family resampling did not improve ranking quality and, at the default threshold, raised false positives; class weights were at least as good.
- A model trained only on ULB scores the 2023 dataset at PR-AUC 0.96, and one trained only on the 2023 data scores all of ULB at 0.53 against a chance level of 0.0017, so the two datasets share fraud signal.

### Run it
```bash
pip install -r requirements.txt
python -m fraud_pipeline.train_ulb        # videos 1 and 2 experiments on ULB (downloads data, about 5 min)
python -m fraud_pipeline.cv_ulb           # 5-fold cross-validation (about 15 min)
python -m fraud_pipeline.train_cc2023     # 2023 dataset model (needs creditcard_2023.csv in data/)
python -m fraud_pipeline.cross_dataset    # transfer between the two datasets
python -m fraud_pipeline.train_paysim     # video 3 dataset (download it into data/ first, see data/README.md)
python -m fraud_pipeline.predict ulb my_transactions.csv
```
Raw data goes in `data/` (git-ignored). Compact trained models are in `models/`, metrics in `results/`.
