"""Re-export churn model artifacts for deployment."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "customer_churn.csv"
ARTIFACTS_PATH = Path(__file__).parent / "churn_model_artifacts.joblib"


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"threshold": threshold, "recall": recall, "precision": precision, "f1": f1, "tp": tp, "fn": fn}


def main():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    data = df.drop(columns=["customerID"]).copy()
    data["Churn"] = (data["Churn"] == "Yes").astype(int)

    data["avg_monthly_spend"] = data["TotalCharges"] / data["tenure"].replace(0, np.nan)
    data["avg_monthly_spend"] = data["avg_monthly_spend"].fillna(data["MonthlyCharges"])
    data["has_internet"] = (data["InternetService"] != "No").astype(int)
    data["long_term_contract"] = data["Contract"].isin(["One year", "Two year"]).astype(int)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "avg_monthly_spend",
        "has_internet",
        "long_term_contract",
    ]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_ap = average_precision_score(y_test, y_proba)

    threshold_grid = np.arange(0.20, 0.71, 0.01)
    threshold_metrics = pd.DataFrame(
        [metrics_at_threshold(y_test, y_proba, t) for t in threshold_grid]
    )
    high_recall = threshold_metrics[threshold_metrics["recall"] >= 0.85]
    if len(high_recall) > 0:
        optimal_row = high_recall.sort_values("f1", ascending=False).iloc[0]
    else:
        optimal_row = threshold_metrics.sort_values("f1", ascending=False).iloc[0]
    optimal_threshold = float(optimal_row["threshold"])

    feature_names = list(numeric_features) + list(
        model.named_steps["prep"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
    )

    artifacts = {
        "model": model,
        "optimal_threshold": optimal_threshold,
        "test_roc_auc": test_roc_auc,
        "test_avg_precision": test_ap,
        "feature_names": feature_names,
        "model_type": "Logistic Regression",
    }

    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"Saved: {ARTIFACTS_PATH}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.2f}")


if __name__ == "__main__":
    main()
