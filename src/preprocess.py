"""
preprocess.py
-------------
Handles all NLP text preprocessing steps.

Why a separate module?
- Separating preprocessing from training is a clean software engineering practice.
- Both the baseline and transformer pipelines can import from a single source of truth.
- Easier to test and modify independently.

NLP Steps Applied:
1. Lowercasing          - Normalizes vocabulary (e.g., "Neural" == "neural")
2. URL/LaTeX removal    - arXiv abstracts often contain LaTeX math and URLs
3. Punctuation removal  - Removes noise; words carry meaning, not symbols
4. Stop-word removal    - Common words ("the", "is") add noise, not signal
5. Lemmatization        - Reduces words to root forms ("running" -> "run")
                          Why Lemmatization over Stemming? Lemmatization uses
                          vocabulary and morphology for accuracy, while stemming
                          is a cruder heuristic. (e.g., "better" -> "good", not "bett")
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources on first run
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)

# ─── Constants ────────────────────────────────────────────────────────────────

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()

# ─── Cleaning Functions ───────────────────────────────────────────────────────

def remove_latex(text: str) -> str:
    r"""Removes common LaTeX math expressions like $x^2$ or \equation."""
    text = re.sub(r"\$[^$]+\$", " ", text)         # Inline math: $...$
    text = re.sub(r"\\\w+\{[^}]*\}", " ", text)    # Commands: \cmd{...}
    text = re.sub(r"\\\w+", " ", text)             # Bare commands: \alpha
    return text


def remove_urls(text: str) -> str:
    """Removes URLs from text."""
    return re.sub(r"http\S+|www\S+", " ", text)


def clean_text(text: str) -> str:
    """
    Full NLP cleaning pipeline.

    Returns a clean, lemmatized, stopword-free string ready for TF-IDF or
    tokenization by a transformer.
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs and LaTeX
    text = remove_urls(text)
    text = remove_latex(text)

    # Step 3: Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)

    # Step 4: Tokenize
    tokens = text.split()

    # Step 5: Remove stopwords and short tokens (len < 3 adds noise)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Step 6: Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "abstract") -> pd.DataFrame:
    """
    Applies the full cleaning pipeline to a DataFrame column.
    Adds a new 'cleaned_abstract' column so the original is preserved.
    """
    print("[+] Preprocessing text (this may take a minute) ...")
    df = df.copy()
    df["cleaned_abstract"] = df[text_col].apply(clean_text)
    print(f"  -> Done. Sample:\n    Original : {df[text_col].iloc[0][:80]}...")
    print(f"    Cleaned  : {df['cleaned_abstract'].iloc[0][:80]}...")
    return df


# ─── Label Encoding ───────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame, label_col: str = "domain"):
    """
    Converts string class labels to integers and returns the mapping.
    Returns: (df with 'label' column, label_to_id dict, id_to_label dict)
    """
    classes     = sorted(df[label_col].unique())
    label_to_id = {c: i for i, c in enumerate(classes)}
    id_to_label = {i: c for c, i in label_to_id.items()}

    df = df.copy()
    df["label"] = df[label_col].map(label_to_id)

    print(f"[+] Label mapping: {label_to_id}")
    return df, label_to_id, id_to_label
