# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
data_collection.py
------------------
Fetches research paper abstracts from the arXiv API and saves them as a CSV.

Why arXiv?
- Real-world, unfiltered data that is far richer than a pre-made Kaggle dataset.
- Abstracts contain dense, domain-specific language, making them ideal for NLP tasks.
- The API is free, stable, and well-documented.

Domains Chosen:
- cs.LG  (Machine Learning)
- cond-mat.mes-hall (Condensed Matter Physics)
- q-bio.QM (Quantitative Biology)
- math.ST (Statistics and Mathematics)
- q-fin.ST (Quantitative Finance)

Why these? They represent distinct scientific vocabularies with minimal overlap,
giving the classifier a stronger signal to learn from.
"""

import arxiv
import pandas as pd
import time
import os
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

DOMAINS = {
    "Machine Learning":          "cs.LG",
    "Condensed Matter Physics":  "cond-mat.mes-hall",
    "Quantitative Biology":      "q-bio.QM",
    "Mathematics":               "math.ST",
    "Quantitative Finance":      "q-fin.ST",
}

PAPERS_PER_DOMAIN = 500        # 500 × 5 domains = 2,500 total abstracts
OUTPUT_DIR        = "data/raw"
OUTPUT_FILE       = os.path.join(OUTPUT_DIR, "abstracts.csv")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_abstracts(domain_label: str, arxiv_category: str, max_results: int) -> list[dict]:
    """
    Queries arXiv for papers in a given category.

    Design Decision: We use the 'Relevance' sort to get a diverse set of
    papers rather than just the most recent ones, which prevents temporal bias.
    """
    print(f"\n[+] Fetching '{domain_label}' (category: {arxiv_category}) ...")

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,     # Politely rate-limit our requests to arXiv
        num_retries=3
    )

    search = arxiv.Search(
        query=f"cat:{arxiv_category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    records = []
    for result in tqdm(client.results(search), total=max_results, desc=f"  {domain_label}"):
        records.append({
            "title":    result.title,
            "abstract": result.summary,
            "domain":   domain_label,
            "arxiv_id": result.entry_id,
        })
        time.sleep(0.1)   # Small additional delay per paper

    print(f"  -> Collected {len(records)} abstracts for '{domain_label}'.")
    return records


def clean_abstract(text: str) -> str:
    """
    Minimal raw-level cleaning at collection time.
    Full NLP preprocessing is done in preprocess.py.

    We only:
    - Strip leading/trailing whitespace
    - Replace newline characters (common in arXiv abstracts)
    """
    text = text.strip()
    text = text.replace("\n", " ")
    return text


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []

    for label, category in DOMAINS.items():
        records = fetch_abstracts(label, category, PAPERS_PER_DOMAIN)

        # Apply minimal cleaning
        for r in records:
            r["abstract"] = clean_abstract(r["abstract"])

        all_records.extend(records)

    df = pd.DataFrame(all_records)

    # Shuffle to prevent any ordering bias when splitting for train/test later
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n[OK] Dataset saved: {OUTPUT_FILE}")
    print(f"   Total samples : {len(df)}")
    print(f"   Class counts  :\n{df['domain'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
