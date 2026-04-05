"""
train_baseline.py
-----------------
Trains a TF-IDF + Logistic Regression baseline model for research domain classification.

Why this is the BASELINE:
- A baseline model establishes a performance floor. Any advanced model must beat it
  to justify its additional complexity and compute cost.
- TF-IDF (Term Frequency-Inverse Document Frequency) captures the importance of a
  word in a document relative to the entire corpus. Words frequent in one domain
  (e.g., "quantum" in Physics) but rare overall get high weights - exactly what we need.
- Logistic Regression is fast, interpretable, and surprisingly competitive on
  high-dimensional text data.

Why NOT just use Naive Bayes?
- Naive Bayes assumes feature independence (all words are independent of each other),
  which is clearly false for language. Logistic Regression makes no such assumption 
  and typically has higher accuracy.
- We'll include both for comparison to show model awareness.

Output:
- models/tfidf_vectorizer.pkl
- models/baseline_lr_model.pkl
- results/baseline_report.txt
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_dataframe, encode_labels

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH    = "data/raw/abstracts.csv"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"
TEST_SIZE    = 0.20   # 80/20 train-test split
RANDOM_STATE = 42

# ─── TF-IDF Configuration ─────────────────────────────────────────────────────
# ngram_range=(1,2): Use both single words AND pairs of words (bigrams).
# Why? "Machine Learning" as a bigram is more informative than "machine" and "learning"
# separately.
# max_features: Cap vocabulary to reduce dimensionality and prevent overfitting.

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 30000,
    "min_df": 2,           # Ignore terms appearing in less than 2 documents
    "sublinear_tf": True,  # Apply log normalization to term frequencies
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Generates and saves a heatmap for the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Confusion matrix saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Data
    print("[1/6] Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  -> {len(df)} samples, {df['domain'].nunique()} classes")

    # 2. Preprocess
    print("[2/6] Preprocessing text ...")
    df = preprocess_dataframe(df, text_col="abstract")
    df, label_to_id, id_to_label = encode_labels(df, label_col="domain")

    X = df["cleaned_abstract"]
    y = df["label"]
    class_names = [id_to_label[i] for i in sorted(id_to_label)]

    # 3. Train/Test Split
    print(f"[3/6] Splitting data ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  -> Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. TF-IDF Vectorization
    print("[4/6] Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"  -> Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Save vectorizer (needed by app.py for inference)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(id_to_label, os.path.join(MODELS_DIR, "id_to_label.pkl"))
    joblib.dump(label_to_id, os.path.join(MODELS_DIR, "label_to_id.pkl"))
    print(f"  -> Vectorizer saved.")

    # 5. Train Models
    print("[5/6] Training models ...")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
    }

    best_model      = None
    best_accuracy   = 0.0
    best_model_name = ""
    results_log     = []

    for name, model in models.items():
        print(f"\n  [*] Training {name} ...")
        model.fit(X_train_tfidf, y_train)

        # Cross-validation on training set for robust accuracy estimate
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="accuracy")
        print(f"     5-fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Evaluate on held-out test set
        y_pred = model.predict(X_test_tfidf)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"     Test Accuracy     : {test_acc:.4f}")

        report = classification_report(y_test, y_pred, target_names=class_names)
        print(f"     Classification Report:\n{report}")

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, class_names,
            title=f"Confusion Matrix - {name}",
            save_path=os.path.join(RESULTS_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
        )

        results_log.append({
            "Model": name,
            "CV Accuracy (mean)": f"{cv_scores.mean():.4f}",
            "CV Accuracy (std)":  f"{cv_scores.std():.4f}",
            "Test Accuracy":      f"{test_acc:.4f}"
        })

        if test_acc > best_accuracy:
            best_accuracy   = test_acc
            best_model      = model
            best_model_name = name

    # 6. Save Best Model
    print(f"\n[6/6] Best model: {best_model_name} (Test Acc: {best_accuracy:.4f})")
    model_path = os.path.join(MODELS_DIR, "baseline_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  -> Model saved: {model_path}")

    # Save a summary report
    summary = pd.DataFrame(results_log)
    report_path = os.path.join(RESULTS_DIR, "baseline_report.csv")
    summary.to_csv(report_path, index=False)
    print(f"  -> Summary report saved: {report_path}")

    print("\n[OK] Baseline training complete!")


if __name__ == "__main__":
    main()
