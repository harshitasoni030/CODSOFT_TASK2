"""Train Logistic Regression, Decision Tree and Random Forest,
compare them on the test file, and save the best full pipeline."""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing import (CATEGORICAL, RAW_FEATURES, TARGET,  # noqa: E402
                           build_preprocessor)

TRAIN_PATH = os.path.join(ROOT, "data", "fraudTrain.csv")
TEST_PATH = os.path.join(ROOT, "data", "fraudTest.csv")
MODEL_PATH = os.path.join(ROOT, "model", "fraud_model.pkl")
METRICS_PATH = os.path.join(ROOT, "model", "metrics.json")
OPTIONS_PATH = os.path.join(ROOT, "model", "options.json")

# Training on 1.3M rows is slow, so we use a stratified sample.
# Set to None to train on every row.
MAX_TRAIN_ROWS = 300_000
RANDOM_STATE = 42


def load(path):
    return pd.read_csv(path, usecols=RAW_FEATURES + [TARGET])


def stratified_sample(df, n):
    if n is None or len(df) <= n:
        return df
    frac = n / len(df)
    parts = [g.sample(frac=frac, random_state=RANDOM_STATE)
             for _, g in df.groupby(TARGET)]
    return pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)


def main():
    print("Loading data...")
    train_df, test_df = load(TRAIN_PATH), load(TEST_PATH)
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")
    print("Train class counts:\n", train_df[TARGET].value_counts().to_string())

    # Save dropdown options for the web form (taken from the real data)
    options = {c: sorted(train_df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    with open(OPTIONS_PATH, "w") as f:
        json.dump(options, f)

    train_df = stratified_sample(train_df, MAX_TRAIN_ROWS)
    X_train, y_train = train_df[RAW_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[RAW_FEATURES], test_df[TARGET]
    print(f"Training on {len(train_df)} rows")

    # class_weight handles the fraud / legitimate imbalance
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
    }

    results, pipelines = {}, {}
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        }
        pipelines[name] = pipe

    table = pd.DataFrame(results).T.round(4)
    print("\n=== Model comparison (on fraudTest.csv) ===")
    print(table.to_string())

    # Accuracy is misleading on imbalanced data, so pick by F1 score
    best_name = table["F1 Score"].idxmax()
    print(f"\nBest model (highest F1): {best_name}")

    joblib.dump(pipelines[best_name], MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
