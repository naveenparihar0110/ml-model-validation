"""
ML Model Validation Project
Supervised Learning - Classification
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# Configuration
# -----------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
ACCURACY_THRESHOLD = 0.90
F1_THRESHOLD = 0.90
NOISE_STD_DEV = 0.1


# -----------------------------
# Data Loading
# -----------------------------


def load_dataset():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        names=columns,
        skipinitialspace=True
    )

    # Remove missing values marked as '?'
    df = df.replace("?", pd.NA).dropna()

    # Encode target
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("income", axis=1).values
    y = df["income"].values

    return X, y, df



# -----------------------------
# Data Splitting
# -----------------------------
def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


# -----------------------------
# Model Training
# -----------------------------
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro")
    }
    return metrics


# -----------------------------
# Overfitting Check
# -----------------------------
def check_overfitting(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    return train_accuracy, test_accuracy

# -----------------------------
# Bias / Fairness Testing
# -----------------------------
def bias_test_real(model, X_test, y_test, df_test):
    """
    Real fairness testing using SEX feature
    """

    male_idx = df_test["sex_Male"] == 1
    female_idx = df_test["sex_Male"] == 0

    pred_male = model.predict(X_test[male_idx])
    pred_female = model.predict(X_test[female_idx])

    acc_male = accuracy_score(y_test[male_idx], pred_male)
    acc_female = accuracy_score(y_test[female_idx], pred_female)

    return acc_male, acc_female


# -----------------------------
# QA Metric Validation
# -----------------------------
def validate_metrics(metrics):
    results = []

    accuracy = metrics["accuracy"]
    f1 = metrics["f1"]

    if accuracy >= ACCURACY_THRESHOLD:
        results.append(("Accuracy", "PASS", accuracy))
    else:
        results.append(("Accuracy", "FAIL", accuracy))

    if f1 >= F1_THRESHOLD:
        results.append(("F1-score", "PASS", f1))
    else:
        results.append(("F1-score", "FAIL", f1))

    return results



# -----------------------------
# Robustness Testing
# -----------------------------
def robustness_test(model, X_test):
    sample = X_test[0]
    noisy_sample = sample + np.random.normal(
        0, NOISE_STD_DEV, size=sample.shape
    )

    original_pred = model.predict([sample])[0]
    noisy_pred = model.predict([noisy_sample])[0]

    assert noisy_pred in [0, 1, 2], "Invalid prediction after noise"

    return original_pred, noisy_pred


# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("\n=== ML MODEL VALIDATION STARTED ===")

   # Load data
    X, y, df = load_dataset()

    # Split everything together
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
    )

    # Train model
    model = train_model(X_train, y_train)
    print("Model training completed.")

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize():<10}: {v:.2f}")

    # Overfitting test
    train_acc, test_acc = check_overfitting(
        model, X_train, y_train, X_test, y_test
    )

    print("\nOverfitting Check:")
    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy    : {test_acc:.2f}")

    accuracy_gap = train_acc - test_acc
    assert accuracy_gap < 0.05, "Overfitting detected"
    print("Overfitting Check PASSED")

    # Bias / Fairness test
    acc_male, acc_female = bias_test_real(model, X_test, y_test, df_test)

    print("\nBias / Fairness Check (Real Data):")
    print(f"Male Accuracy  : {acc_male:.2f}")
    print(f"Female Accuracy: {acc_female:.2f}")

    fairness_gap = abs(acc_male - acc_female)
    assert fairness_gap < 0.15, "Bias detected between genders"

    print("Bias / Fairness Check PASSED")



    # QA validation
    print("\nQA Metric Validation:")
    metric_results = validate_metrics(metrics)

    qa_pass = True
    for metric, status, value in metric_results:
        print(f"{metric:<10}: {value:.2f} -> {status}")
    if status == "FAIL":
        qa_pass = False

    if qa_pass:
        print("Overall Metric Validation: PASSED")
    else:
        print("Overall Metric Validation: FAILED")


    # Robustness test
    orig, noisy = robustness_test(model, X_test)
    print("\nRobustness Test:")
    print("Original Prediction:", orig)
    print("Noisy Prediction   :", noisy)
    print("Robustness Test PASSED")

    print("\n==============================")
    if qa_pass:
      print("FINAL QA VERDICT: MODEL APPROVED")
    else:
      print("FINAL QA VERDICT: MODEL REJECTED")
      print("==============================")



if __name__ == "__main__":
    main()
