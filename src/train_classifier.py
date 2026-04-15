import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocess import (
    load_data,
    replace_question_marks,
    clean_label,
    split_features_target_weight,
    detect_feature_types,
    build_preprocessor
)


def evaluate_model(model_name, y_test, y_pred, y_prob):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print(f"\n{'=' * 60}")
    print(f"{model_name} Results")
    print(f"{'=' * 60}")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def main():
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    print("Loading data...")
    df = load_data("data/census-bureau.data", "data/census-bureau.columns")

    print("Cleaning data...")
    df = replace_question_marks(df)
    df = clean_label(df)

    X, y, w = split_features_target_weight(df)
    numeric_cols, categorical_cols = detect_feature_types(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    print(f"Number of numeric columns: {len(numeric_cols)}")
    print(f"Number of categorical columns: {len(categorical_cols)}")

    print("Splitting data into train and test...")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    all_metrics = []

    print("\nTraining Logistic Regression...")
    log_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    log_model.fit(X_train, y_train, model__sample_weight=w_train)
    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]

    log_metrics = evaluate_model("Logistic Regression", y_test, y_pred_log, y_prob_log)
    all_metrics.append(log_metrics)

    joblib.dump(log_model, "outputs/models/logistic_model.joblib")

    print("\nTraining Random Forest...")
    rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    rf_model.fit(X_train, y_train, model__sample_weight=w_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    rf_metrics = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)
    all_metrics.append(rf_metrics)

    joblib.dump(rf_model, "outputs/models/random_forest_model.joblib")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("outputs/tables/classification_metrics.csv", index=False)

    print("\nSaved results to:")
    print("- outputs/models/logistic_model.joblib")
    print("- outputs/models/random_forest_model.joblib")
    print("- outputs/tables/classification_metrics.csv")


if __name__ == "__main__":
    main()