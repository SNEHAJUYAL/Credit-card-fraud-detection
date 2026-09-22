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
