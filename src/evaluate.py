"""
evaluate.py
-----------
Loads all trained models and generates a comprehensive comparison report
with confusion matrices and accuracy comparison charts.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_dataframe, encode_labels

# --- Configuration -----------------------------------------------------------

DATA_PATH        = "data/raw/abstracts.csv"
BASELINE_MODEL   = "models/baseline_model.pkl"
VECTORIZER_PATH  = "models/tfidf_vectorizer.pkl"
BERT_FEATURES    = "models/distilbert_features.npz"
BERT_CLASSIFIER  = "models/distilbert_classifier.pkl"
ID_TO_LABEL_PATH = "models/id_to_label.pkl"
RESULTS_DIR      = "results"
TEST_SIZE        = 0.20
RANDOM_STATE     = 42

# --- Helpers -----------------------------------------------------------------

def plot_confusion_matrix(cm, class_names, title, save_path, cmap="Blues"):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_model_comparison(results_df, save_path):
    colors = ["#4C72B0", "#55A868", "#DD8452"][:len(results_df)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(results_df["Model"], results_df["Test Accuracy"],
                   color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Test Accuracy", fontsize=12)
    ax.set_title("Model Comparison - Research Domain Classification", fontsize=14, fontweight="bold")
    for bar, acc in zip(bars, results_df["Test Accuracy"]):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.4f}", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved: {save_path}")


# --- Main --------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load & prepare data
    print("[1/4] Loading & preprocessing dataset ...")
    df = pd.read_csv(DATA_PATH)
    df = preprocess_dataframe(df, text_col="abstract")
    df, label_to_id, id_to_label = encode_labels(df, label_col="domain")
    class_names = [id_to_label[i] for i in sorted(id_to_label)]

    X_clean = df["cleaned_abstract"].tolist()
    y_all   = df["label"].tolist()

    _, X_test_clean, _, y_test = train_test_split(
        X_clean, y_all,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )

    comparison_results = []

    # 2. Evaluate Baseline (TF-IDF + LR)
    if os.path.exists(BASELINE_MODEL) and os.path.exists(VECTORIZER_PATH):
        print("\n[2/4] Evaluating Baseline (TF-IDF + Logistic Regression) ...")
        vectorizer = joblib.load(VECTORIZER_PATH)
        model      = joblib.load(BASELINE_MODEL)
        X_tfidf    = vectorizer.transform(X_test_clean)
        y_pred     = model.predict(X_tfidf)
        acc        = accuracy_score(y_test, y_pred)
        report     = classification_report(y_test, y_pred, target_names=class_names)
        cm         = confusion_matrix(y_test, y_pred)

        print(f"  Test Accuracy : {acc:.4f}")
        print(f"\n{report}")

        plot_confusion_matrix(cm, class_names,
            title="Confusion Matrix - TF-IDF + Logistic Regression (Baseline)",
            save_path=os.path.join(RESULTS_DIR, "eval_cm_baseline.png"),
            cmap="Blues")

        with open(os.path.join(RESULTS_DIR, "eval_baseline_report.txt"), "w") as f:
            f.write(f"Model: TF-IDF + Logistic Regression\n")
            f.write(f"Test Accuracy: {acc:.4f}\n\n{report}")

        comparison_results.append({"Model": "TF-IDF + LR (Baseline)", "Test Accuracy": acc})
    else:
        print("[!] Baseline model not found. Run train_baseline.py first.")

    # 3. Evaluate BERT Features + LR
    if os.path.exists(BERT_FEATURES) and os.path.exists(BERT_CLASSIFIER):
        print("\n[3/4] Evaluating DistilBERT Features + LR ...")
        data   = np.load(BERT_FEATURES)
        X_bert = data["embeddings"]
        y_bert = data["labels"]

        _, X_test_bert, _, y_test_bert = train_test_split(
            X_bert, y_bert,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bert
        )

        clf    = joblib.load(BERT_CLASSIFIER)
        y_pred = clf.predict(X_test_bert)
        acc    = accuracy_score(y_test_bert, y_pred)
        report = classification_report(y_test_bert, y_pred, target_names=class_names)
        cm     = confusion_matrix(y_test_bert, y_pred)

        print(f"  Test Accuracy : {acc:.4f}")
        print(f"\n{report}")

        plot_confusion_matrix(cm, class_names,
            title="Confusion Matrix - DistilBERT Features + LR",
            save_path=os.path.join(RESULTS_DIR, "eval_cm_transformer.png"),
            cmap="Purples")

        with open(os.path.join(RESULTS_DIR, "eval_transformer_report.txt"), "w") as f:
            f.write(f"Model: DistilBERT Features + Logistic Regression\n")
            f.write(f"Test Accuracy: {acc:.4f}\n\n{report}")

        comparison_results.append({"Model": "DistilBERT Features + LR", "Test Accuracy": acc})
    else:
        print("[!] BERT model not found. Run train_transformer.py first.")

    # 4. Generate comparison chart
    if comparison_results:
        print("\n[4/4] Generating model comparison chart ...")
        comp_df = pd.DataFrame(comparison_results)
        comp_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
        plot_model_comparison(comp_df,
            save_path=os.path.join(RESULTS_DIR, "model_comparison.png"))
        print("\n=== FINAL RESULTS ===")
        print(comp_df.to_string(index=False))

    print("\n[OK] Evaluation complete! Charts saved to results/")


if __name__ == "__main__":
    main()
