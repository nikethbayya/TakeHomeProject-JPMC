import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(data_path, columns_path):
    with open(columns_path, "r", encoding="utf-8") as f:
        columns = [line.strip() for line in f if line.strip()]

    df = pd.read_csv(data_path, header=None, names=columns)
    return df


def replace_question_marks(df):
    return df.replace("?", np.nan)


def clean_label(df):
    df["label"] = df["label"].astype(str).str.strip()

    label_map = {
        "- 50000.": 0,
        "50000+.": 1
    }

    df["label"] = df["label"].map(label_map)
    return df


def split_features_target_weight(df):
    y = df["label"]
    w = df["weight"]
    X = df.drop(columns=["label", "weight"])
    return X, y, w


def detect_feature_types(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor


if __name__ == "__main__":
    df = load_data("data/census-bureau.data", "data/census-bureau.columns")
    print("Shape before cleaning:", df.shape)
    print(df.head())

    df = replace_question_marks(df)
    df = clean_label(df)

    print("\nLabel counts:")
    print(df["label"].value_counts(dropna=False))

    print("\nMissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))