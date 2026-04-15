import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocess import (
    load_data,
    replace_question_marks,
    clean_label,
    detect_feature_types,
    build_preprocessor
)


def mode_or_na(series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def main():
    os.makedirs("outputs/tables", exist_ok=True)

    print("Loading data...")
    df = load_data("data/census-bureau.data", "data/census-bureau.columns")

    print("Cleaning data...")
    df = replace_question_marks(df)
    df = clean_label(df)

    print("Preparing data for segmentation...")
    X_seg = df.drop(columns=["label", "weight"])

    numeric_cols, categorical_cols = detect_feature_types(X_seg)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    print(f"Number of numeric columns: {len(numeric_cols)}")
    print(f"Number of categorical columns: {len(categorical_cols)}")

    print("Transforming features...")
    X_processed = preprocessor.fit_transform(X_seg)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    print("Running PCA...")
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_processed)

    print("Testing different numbers of clusters...")
    cluster_results = []
    best_k = None
    best_score = -1

    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, cluster_labels)

        cluster_results.append({
            "k": k,
            "silhouette_score": score
        })

        print(f"k={k}, silhouette_score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    cluster_results_df = pd.DataFrame(cluster_results)
    cluster_results_df.to_csv("outputs/tables/cluster_scores.csv", index=False)

    print(f"\nBest number of clusters based on silhouette score: {best_k}")

    print("Fitting final K-Means model...")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = final_kmeans.fit_predict(X_pca)

    print("Building cluster summary...")
    cluster_summary = df.groupby("cluster").agg({
        "age": "mean",
        "wage per hour": "mean",
        "capital gains": "mean",
        "capital losses": "mean",
        "dividends from stocks": "mean",
        "weeks worked in year": "mean",
        "education": mode_or_na,
        "marital stat": mode_or_na,
        "major occupation code": mode_or_na,
        "sex": mode_or_na,
        "race": mode_or_na,
        "label": "mean"
    }).reset_index()

    cluster_summary = cluster_summary.rename(columns={
        "label": "pct_income_gt_50k"
    })

    cluster_sizes = df["cluster"].value_counts().sort_index().reset_index()
    cluster_sizes.columns = ["cluster", "count"]

    cluster_summary = cluster_summary.merge(cluster_sizes, on="cluster", how="left")

    cluster_summary.to_csv("outputs/tables/cluster_summary.csv", index=False)
    df.to_csv("outputs/tables/segmented_data.csv", index=False)

    print("\nSaved results to:")
    print("- outputs/tables/cluster_scores.csv")
    print("- outputs/tables/cluster_summary.csv")
    print("- outputs/tables/segmented_data.csv")

    print("\nCluster Summary:")
    print(cluster_summary)


if __name__ == "__main__":
    main()