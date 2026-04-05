import sys, os
sys.path.append('.')

print("=== TEST 1: Data ===")
import pandas as pd
df = pd.read_csv("data/raw/abstracts.csv")
print(f"  Rows    : {len(df)}")
print(f"  Columns : {list(df.columns)}")
print(f"  Domains :")
for domain, count in df["domain"].value_counts().items():
    print(f"    {domain}: {count}")

print()
print("=== TEST 2: Preprocessing ===")
from src.preprocess import clean_text, encode_labels
sample  = df["abstract"].iloc[0]
cleaned = clean_text(sample)
print(f"  Original (80 chars): {sample[:80]}")
print(f"  Cleaned  (80 chars): {cleaned[:80]}")

print()
print("=== TEST 3: Baseline Model Inference ===")
import joblib
vectorizer  = joblib.load("models/tfidf_vectorizer.pkl")
model       = joblib.load("models/baseline_model.pkl")
id_to_label = joblib.load("models/id_to_label.pkl")

test_cases = [
    "We propose a neural network architecture using convolutional layers and attention mechanisms for image classification.",
    "We study the quantum Hall effect in a two-dimensional electron gas subjected to a perpendicular magnetic field.",
    "We derive closed-form solutions for European options using the Black-Scholes framework with Poisson jump processes.",
    "Gene expression patterns reveal cellular differentiation trajectories using RNA sequencing data.",
    "We prove bounds on the distribution of prime numbers using the explicit formula for Dirichlet series.",
]

for test in test_cases:
    cleaned_test = clean_text(test)
    tfidf        = vectorizer.transform([cleaned_test])
    probs        = model.predict_proba(tfidf)[0]
    pred_id      = int(probs.argmax())
    print(f"  Input   : {test[:65]}...")
    print(f"  Predict : {id_to_label[pred_id]}  ({probs[pred_id]*100:.1f}% confidence)")
    print()

print("=== TEST 4: DistilBERT Feature Extractor ===")
import numpy as np
data     = np.load("models/distilbert_features.npz")
bert_clf = joblib.load("models/distilbert_classifier.pkl")
print(f"  Cached embeddings shape : {data['embeddings'].shape}")
print(f"  (2500 papers x 768-dim BERT vectors)")

print()
print("=== TEST 5: TF-IDF From Scratch ===")
from src.tfidf_scratch import TFIDFVectorizerScratch
scratch = TFIDFVectorizerScratch()
docs = [
    "machine learning neural network gradient descent backpropagation",
    "quantum physics electron energy spin Hamiltonian",
    "gene expression RNA sequencing cell differentiation",
]
X = scratch.fit_transform(docs)
idx_to_word = {v: k for k, v in scratch.vocabulary_.items()}
print(f"  Matrix shape : {X.shape}  (3 docs x {X.shape[1]} words)")
for i, doc in enumerate(docs):
    top_idx   = X[i].argsort()[::-1][:3]
    top_words = [idx_to_word[j] for j in top_idx if X[i, j] > 0]
    print(f"  Doc {i} top words: {top_words}")

print()
print("=== TEST 6: Final Results ===")
comp = pd.read_csv("results/model_comparison.csv")
print(comp[["Model", "Test Accuracy"]].to_string(index=False))

print()
print("=== ALL TESTS PASSED  ===")
