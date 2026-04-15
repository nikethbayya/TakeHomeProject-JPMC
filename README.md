# Take Home Project
## Census Income Classification and Customer Segmentation

**Objective:**

> **(1)** Predict whether a person belongs to the income group above \$50,000 or at or below \$50,000, and  
> **(2)** Create customer segments that can support a more targeted marketing strategy.

This repository contains code for data loading, preprocessing, classification modeling, segmentation modeling, and result generation.


## ✨ What this project does

This project addresses two related analytics tasks using demographic and employment-related census data.

### 1) Income Classification
A supervised learning pipeline that predicts whether an individual's income is:

- **0** → at or below \$50,000
- **1** → above \$50,000

Two classification models are trained and evaluated:

- **Logistic Regression**
- **Random Forest**

### 2) Customer Segmentation
An unsupervised learning pipeline that groups individuals into meaningful customer profiles for marketing use.

The segmentation workflow uses:

- preprocessing for mixed numeric and categorical data
- **PCA** for dimensionality reduction
- **K-Means clustering**
- silhouette score to compare candidate cluster counts

---

## 🧰 Tech Stack

- **Python**
- **pandas** — data loading and transformation
- **numpy** — numerical operations
- **scikit-learn** — preprocessing, classification, PCA, clustering, evaluation
- **matplotlib** — plotting support
- **seaborn** — data visualization support
- **joblib** — saving trained models

---

## 📁 Project Structure

```text
TakeHomeProject-JPMC/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── preprocess.py
│   ├── train_classifier.py
│   └── segment_customers.py
└── outputs/
    └── tables/
```

## 🚀 Run Locally

**1. Create a virtual environment
macOS / Linux**

```bash
python3 -m venv census
source census/bin/activate
```

**Windows**

```bash
python -m venv census
census\Scripts\activate
```
**2. Install dependencies**

```bash
pip install -r requirements.txt
```

## 📋 Requirements
The project dependencies are listed in ```requirements.txt```:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

## ▶️ How to Execute the Code
Run all commands from the project root directory.

**Step 1: Preprocess and inspect the dataset**

This script:

• loads the dataset

• assigns column names

• replaces ```?``` with null values

• converts the target label into binary form

• prints dataset shape, label counts, and missing-value summary

```bash
python src/preprocess.py
```

**Step 2: Train and evaluate the classification models**

This script:

• loads and cleans the data

• separates features, target, and sample weight

• builds a preprocessing pipeline

• trains Logistic Regression

• trains Random Forest

• evaluates both models on a held-out test set

• saves trained models and classification metrics

```bash
python src/train_classifier.py
```

**Step 3: Generate the customer segmentation model**

This script:

• loads and cleans the data

• removes ```label``` and ```weight``` from clustering inputs

• preprocesses numeric and categorical features

• applies PCA for dimensionality reduction

• evaluates K-Means clustering across multiple values of ```k```

• selects the best cluster count using the silhouette score

• saves cluster scores, cluster summaries, and row-level segment assignments

```bash
python src/segment_customers.py
```
## 📊 Output Files

Running the scripts generates outputs in the ```outputs/``` directory.

**Classification outputs**

• ```outputs/models/logistic_model.joblib```

• ```outputs/models/random_forest_model.joblib```

• ```outputs/tables/classification_metrics.csv```

**Segmentation outputs**

• ```outputs/tables/cluster_scores.csv```

• ```outputs/tables/cluster_summary.csv```

• ```outputs/tables/segmented_data.csv```



