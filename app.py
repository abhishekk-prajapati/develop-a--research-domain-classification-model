"""
app.py
------
Streamlit web application for Research Domain Classification.
Run with: streamlit run app.py
"""

import os
import sys
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Domain Classifier",

    page_icon="[RESEARCH]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header { text-align: center; padding: 2rem 0 1rem; }
    .main-header h1 {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .main-header p { font-size: 1.1rem; color: #6b7280; margin-top: 0.5rem; }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 2rem; color: white;
        text-align: center; margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    .prediction-card h2 { font-size: 1.4rem; font-weight: 500; opacity: 0.9; margin-bottom: 0.5rem; }
    .prediction-card h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }

    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .metric-card h3 {
        color: #475569; font-size: 0.85rem; text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 0.5rem;
    }
    .metric-card p { font-size: 1.8rem; font-weight: 700; color: #1e293b; margin: 0; }

    .stTextArea textarea {
        border-radius: 12px !important; border: 2px solid #e2e8f0 !important;
        font-size: 0.95rem !important;
    }
    .stButton > button {
        width: 100%; padding: 0.75rem 2rem; border-radius: 10px; font-weight: 600;
        font-size: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; transition: all 0.2s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5);
        transform: translateY(-1px);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
    }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "Machine Learning":         "#667eea",
    "Condensed Matter Physics": "#f093fb",
    "Quantitative Biology":     "#4facfe",
    "Mathematics":              "#43e97b",
    "Quantitative Finance":     "#fa709a",
}

DOMAIN_ICONS = {
    "Machine Learning":         "[ML]",
    "Condensed Matter Physics": "[PHY]",
    "Quantitative Biology":     "[BIO]",
    "Mathematics":              "[MATH]",
    "Quantitative Finance":     "[FIN]",
}

EXAMPLES = {
    "[ML] Machine Learning": (
        "We propose a novel attention-based transformer architecture for "
        "multi-label text classification. Our model employs self-attention "
        "mechanisms to capture long-range dependencies between tokens and "
        "achieves state-of-the-art performance on multiple benchmark datasets. "
        "We evaluate on the standard train/test splits and report F1 scores."
    ),
    "[PHY] Physics": (
        "We study the quantum Hall effect in a two-dimensional electron gas "
        "subjected to a perpendicular magnetic field. Using the Landau level "
        "formalism, we derive the Hall conductance as a topological invariant "
        "and demonstrate its robustness against disorder and impurities."
    ),
    "[BIO] Biology": (
        "We present a computational framework for analyzing single-cell RNA "
        "sequencing data to identify cell-type-specific gene expression patterns. "
        "Our method leverages dimensionality reduction and clustering algorithms "
        "to reconstruct cellular differentiation trajectories from noisy measurements."
    ),
    "[MATH] Mathematics": (
        "We prove a generalization of the Riemann hypothesis for certain L-functions "
        "attached to automorphic forms. Our proof employs techniques from analytic "
        "number theory including the explicit formula and zero-free regions for "
        "Dirichlet series. We derive bounds on the distribution of prime numbers."
    ),
    "[FIN] Quantitative Finance": (
        "We propose a stochastic volatility model for option pricing under jump "
        "diffusion. Using the Black-Scholes framework extended with Poisson processes, "
        "we derive closed-form solutions for European put and call options. The model "
        "is calibrated to market data using maximum likelihood estimation."
    ),
}

# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource
def load_baseline_model():
    """Load TF-IDF + Logistic Regression. Cached — loads only once."""
    try:
        vectorizer  = joblib.load("models/tfidf_vectorizer.pkl")
        model       = joblib.load("models/baseline_model.pkl")
        id_to_label = joblib.load("models/id_to_label.pkl")
        return vectorizer, model, id_to_label
    except FileNotFoundError:
        return None, None, None


@st.cache_resource
def load_distilbert_resources():
    """Load DistilBERT tokenizer + model for feature extraction. Cached."""
    try:
        tokenizer   = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        bert_model  = DistilBertModel.from_pretrained("distilbert-base-uncased")
        bert_model.eval()
        clf         = joblib.load("models/distilbert_classifier.pkl")
        id_to_label = joblib.load("models/id_to_label.pkl")
        return tokenizer, bert_model, clf, id_to_label
    except (FileNotFoundError, OSError):
        return None, None, None, None


# ─── Prediction Functions ─────────────────────────────────────────────────────

def predict_baseline(abstract, vectorizer, model, id_to_label):
    cleaned  = clean_text(abstract)
    tfidf    = vectorizer.transform([cleaned])
    probs    = model.predict_proba(tfidf)[0]
    label_id = int(probs.argmax())
    labels   = [id_to_label[i] for i in range(len(probs))]
    return id_to_label[label_id], probs, labels


def predict_distilbert(abstract, tokenizer, bert_model, clf, id_to_label):
    encoding = tokenizer(
        abstract, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        output = bert_model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
    cls_emb  = output.last_hidden_state[:, 0, :].numpy()
    probs    = clf.predict_proba(cls_emb)[0]
    label_id = int(probs.argmax())
    labels   = [id_to_label[i] for i in range(len(probs))]
    return id_to_label[label_id], probs, labels


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")
    selected_model = st.selectbox(
        "Select Model",
        ["TF-IDF + Logistic Regression (Baseline)", "DistilBERT Features + LR"],
        help="Baseline is faster. DistilBERT uses deep semantic embeddings."
    )
    st.markdown("---")
    st.markdown("## About This Project")
    st.markdown("""
A **Research Domain Classifier** built with real NLP techniques.

**Pipeline:**
1. Data scraped from arXiv API (2,500 abstracts)
2. Text cleaned via NLP preprocessing (lemmatization, TF-IDF)
3. Two models trained and compared
4. Deployed as an interactive Streamlit app

**Domains:**
- Machine Learning
- Condensed Matter Physics
- Quantitative Biology
- Mathematics
- Quantitative Finance
    """)
    st.markdown("---")
    st.markdown("## Try an Example")
    selected_example = st.selectbox(
        "Load a sample abstract",
        ["(Select an example)"] + list(EXAMPLES.keys())
    )

# ─── Main Content ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>Research Domain Classifier</h1>
    <p>Paste a research paper abstract and let AI classify its scientific domain.</p>
</div>
""", unsafe_allow_html=True)

default_text = ""
if selected_example != "(Select an example)":
    default_text = EXAMPLES[selected_example]

col1, col2 = st.columns([2, 1])

with col1:
    abstract_input = st.text_area(
        "Enter a Research Abstract",
        value=default_text,
        height=220,
        placeholder="Paste any research paper abstract here...",
        key="abstract_input"
    )
    classify_btn = st.button("Classify Domain", key="classify_btn")

with col2:
    st.markdown("### Domain Reference")
    for domain, icon in DOMAIN_ICONS.items():
        color = DOMAIN_COLORS.get(domain, "#667eea")
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin:0.4rem 0;
                    padding:0.5rem 0.8rem; border-radius:8px;
                    background:{color}20; border-left:3px solid {color};">
            <span style="font-size:1rem; font-weight:600; color:#1e293b;
                         margin-right:0.5rem;">{icon}</span>
            <span style="font-size:0.9rem; color:#1e293b;">{domain}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Classification ───────────────────────────────────────────────────────────

if classify_btn:
    if not abstract_input.strip():
        st.warning("Please enter an abstract before classifying.")
    elif len(abstract_input.strip()) < 50:
        st.warning("Abstract seems too short. Please enter at least a few sentences.")
    else:
        predicted_label = None
        probs = None
        class_names = None

        with st.spinner("Classifying..."):
            if "Baseline" in selected_model:
                vectorizer, model, id_to_label = load_baseline_model()
                if model is None:
                    st.error("Baseline model not found. Please run: python src/train_baseline.py")
                else:
                    predicted_label, probs, class_names = predict_baseline(
                        abstract_input, vectorizer, model, id_to_label
                    )
            else:
                tokenizer, bert_model, clf, id_to_label = load_distilbert_resources()
                if clf is None:
                    st.error("DistilBERT model not found. Please run: python src/train_transformer.py")
                else:
                    with st.spinner("Extracting semantic embeddings from DistilBERT..."):
                        predicted_label, probs, class_names = predict_distilbert(
                            abstract_input, tokenizer, bert_model, clf, id_to_label
                        )

        if predicted_label and probs is not None:
            icon       = DOMAIN_ICONS.get(predicted_label, "[?]")
            confidence = float(max(probs)) * 100

            st.markdown(f"""
            <div class="prediction-card">
                <h2>Predicted Domain</h2>
                <h1>{icon} {predicted_label}</h1>
                <p style="opacity:0.85; margin-top:0.5rem; font-size:1rem;">
                    Confidence: <strong>{confidence:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Confidence Scores (All Domains)")
            sorted_idx    = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            sorted_labels = [class_names[i] for i in sorted_idx]
            sorted_probs  = [float(probs[i]) * 100 for i in sorted_idx]
            sorted_colors = [DOMAIN_COLORS.get(lbl, "#667eea") for lbl in sorted_labels]

            fig = go.Figure(go.Bar(
                x=sorted_probs, y=sorted_labels, orientation="h",
                marker=dict(color=sorted_colors, line=dict(width=0)),
                text=[f"{p:.1f}%" for p in sorted_probs],
                textposition="outside"
            ))
            fig.update_layout(
                margin=dict(l=0, r=60, t=10, b=10), height=280,
                xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#f0f0f0",
                           title="Confidence (%)"),
                yaxis=dict(tickfont=dict(size=12)),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

# ─── Model Comparison ─────────────────────────────────────────────────────────

metrics_path = "results/model_comparison.csv"
if os.path.exists(metrics_path):
    st.markdown("---")
    st.markdown("### Model Performance Comparison")

    comp_df = pd.read_csv(metrics_path)
    cols = st.columns(len(comp_df))
    for i, (_, row) in enumerate(comp_df.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{row['Model']}</h3>
                <p>{float(row['Test Accuracy'])*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    fig2 = px.bar(
        comp_df, x="Model", y="Test Accuracy",
        color="Model", color_discrete_sequence=["#667eea", "#764ba2"],
        title="Model Accuracy Comparison", text="Test Accuracy",
    )
    fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig2.update_layout(
        yaxis=dict(range=[0, 1.1], title="Test Accuracy"),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white", height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)
