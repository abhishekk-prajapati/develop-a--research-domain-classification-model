"""
train_transformer.py
--------------------
Uses DistilBERT as a frozen feature extractor, then trains a Logistic Regression
classifier on top. This approach runs in ~3 minutes on CPU.

Why this approach instead of full fine-tuning?
- Full fine-tuning DistilBERT requires GPU to be practical (56+ minutes on CPU).
- Feature extraction: we freeze ALL DistilBERT weights, extract [CLS] embeddings
  (768-dim vectors that encode semantic meaning), then train a simple LR head.
- This still demonstrates real Transformer / NLP skills and produces strong results.
- It is a valid, production-used technique (e.g., OpenAI embeddings + LR head).

Design Decision:
  We use the [CLS] token embedding as the sentence representation.
  Why [CLS]? In BERT-style models, the [CLS] token is prepended to every input
  and its final hidden state is trained to aggregate the full-sequence meaning.
  It is the standard representation used for classification tasks.

Output:
  - models/distilbert_features.npz   (saved embeddings, so re-extraction is skipped)
  - models/distilbert_classifier.pkl (LR trained on BERT features)
  - results/transformer_report.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import encode_labels

# --- Configuration -----------------------------------------------------------

DATA_PATH        = "data/raw/abstracts.csv"
MODELS_DIR       = "models"
RESULTS_DIR      = "results"
FEATURES_PATH    = os.path.join(MODELS_DIR, "distilbert_features.npz")
CLASSIFIER_PATH  = os.path.join(MODELS_DIR, "distilbert_classifier.pkl")

MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 128    # Shorter for speed; abstracts are content-dense up front
BATCH_SIZE   = 32
TEST_SIZE    = 0.20
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Feature Extraction ------------------------------------------------------

def extract_cls_embeddings(texts, tokenizer, model, batch_size, max_length, device):
    """
    Passes text through DistilBERT and returns the [CLS] token embedding
    for each input sentence. No gradients are computed (model is frozen).
    """
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Extracting embeddings"):
        batch_texts = texts[i : i + batch_size]

        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state shape: (batch, seq_len, 768)
        # [CLS] token is always at index 0
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix - DistilBERT Features + LR", fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Confusion matrix saved: {save_path}")


# --- Main --------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")

    # 1. Load data
    print("\n[1/5] Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    df, label_to_id, id_to_label = encode_labels(df, label_col="domain")
    class_names = [id_to_label[i] for i in sorted(id_to_label)]

    texts  = df["abstract"].tolist()
    labels = np.array(df["label"].tolist())

    # 2. Extract features (or load cached)
    if os.path.exists(FEATURES_PATH):
        print("\n[2/5] Loading cached BERT embeddings ...")
        data = np.load(FEATURES_PATH)
        X    = data["embeddings"]
        y    = data["labels"]
        print(f"  -> Loaded {X.shape[0]} embeddings of dim {X.shape[1]}")
    else:
        print(f"\n[2/5] Loading DistilBERT ({MODEL_NAME}) ...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        model     = DistilBertModel.from_pretrained(MODEL_NAME).to(DEVICE)

        print(f"  Extracting [CLS] embeddings for {len(texts)} abstracts ...")
        print(f"  Batch size: {BATCH_SIZE} | Max length: {MAX_LENGTH} tokens")
        X = extract_cls_embeddings(texts, tokenizer, model, BATCH_SIZE, MAX_LENGTH, DEVICE)
        y = labels

        np.savez(FEATURES_PATH, embeddings=X, labels=y)
        print(f"  -> Features cached: {FEATURES_PATH}  ({X.shape})")
        del model  # Free memory

    # 3. Train/test split
    print("\n[3/5] Splitting data ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  -> Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Train classifier on BERT features
    print("\n[4/5] Training Logistic Regression on BERT features ...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    y_pred   = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=class_names)

    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # 5. Save
    print("[5/5] Saving model and results ...")
    joblib.dump(clf, CLASSIFIER_PATH)
    print(f"  -> Classifier saved: {CLASSIFIER_PATH}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names,
                          save_path=os.path.join(RESULTS_DIR, "eval_cm_transformer.png"))

    summary = pd.DataFrame([{
        "Model": "DistilBERT Features + LR",
        "CV Accuracy (mean)": f"{cv_scores.mean():.4f}",
        "CV Accuracy (std)":  f"{cv_scores.std():.4f}",
        "Test Accuracy":      f"{test_acc:.4f}"
    }])
    summary.to_csv(os.path.join(RESULTS_DIR, "transformer_report.csv"), index=False)

    # Merge with baseline report for comparison
    baseline_csv = os.path.join(RESULTS_DIR, "baseline_report.csv")
    if os.path.exists(baseline_csv):
        baseline_df = pd.read_csv(baseline_csv)
        combined    = pd.concat([baseline_df, summary], ignore_index=True)
        combined["Test Accuracy"] = combined["Test Accuracy"].astype(float)
        combined.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
        print("\n  Model Comparison:")
        print(combined[["Model", "Test Accuracy"]].to_string(index=False))

    print("\n[OK] Transformer feature extraction + training complete!")


if __name__ == "__main__":
    main()
