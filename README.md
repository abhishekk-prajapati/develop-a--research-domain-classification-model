# Research Domain Classification using NLP

> **An end-to-end Machine Learning / NLP project** that classifies research paper abstracts into scientific domains using both traditional Machine Learning and modern Deep Learning (Transformers) approaches.

---

## 🎯 Project Overview

Scientific literature is growing exponentially. Automated classification of research abstracts is a valuable real-world NLP task that touches on:
- **Data Engineering** (collecting raw data from the arXiv API)
- **NLP Preprocessing** (text cleaning, lemmatization, TF-IDF)
- **Machine Learning** (Logistic Regression, Naive Bayes)
- **Deep Learning / NLP** (fine-tuning DistilBERT)
- **Model Evaluation** (confusion matrices, F1, cross-validation)
- **Deployment** (interactive Streamlit web app)

### Domains Classified
| Icon | Domain | arXiv Category |
|:---:|:---|:---|
| 🤖 | Machine Learning | cs.LG |
| ⚛️ | Condensed Matter Physics | cond-mat.mes-hall |
| 🧬 | Quantitative Biology | q-bio.QM |
| 📐 | Mathematics | math.ST |
| 📈 | Quantitative Finance | q-fin.ST |

---

## 📁 Project Structure

```
research-domain-classifier/
│
├── data/
│   └── raw/
│       └── abstracts.csv          # Raw dataset (~2,500 abstracts from arXiv API)
│
├── notebooks/
│   └── 01_EDA.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── data_collection.py         # arXiv API scraper
│   ├── preprocess.py              # NLP preprocessing utilities
│   ├── train_baseline.py          # TF-IDF + Logistic Regression
│   ├── train_transformer.py       # DistilBERT fine-tuning
│   └── evaluate.py                # Model comparison & evaluation
│
├── models/
│   ├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
│   ├── baseline_model.pkl         # Saved baseline model
│   └── distilbert_classifier/     # Saved DistilBERT weights
│
├── results/
│   ├── model_comparison.png       # Bar chart comparing model accuracies
│   ├── eval_cm_baseline.png       # Confusion matrix for baseline
│   └── eval_cm_transformer.png    # Confusion matrix for transformer
│
├── app.py                         # Streamlit web app
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🔧 Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/research-domain-classifier.git
cd research-domain-classifier

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (auto-runs on first use, or run manually)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 🚀 Running the Project (Step-by-Step)

### Step 1: Collect Data from arXiv
```bash
python src/data_collection.py
```
This fetches ~2,500 research abstracts from the arXiv API and saves them to `data/raw/abstracts.csv`.

### Step 2: Explore the Data (Optional but Recommended)
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Step 3: Train the Baseline Model
```bash
python src/train_baseline.py
```
Trains TF-IDF + Logistic Regression & Naive Bayes, compares them, and saves the best model.

### Step 4: Fine-tune the Transformer Model
```bash
python src/train_transformer.py
```
Fine-tunes DistilBERT for sequence classification. *Note: GPU recommended; CPU will work but takes ~30–60 minutes.*

### Step 5: Evaluate & Compare Models
```bash
python src/evaluate.py
```
Generates confusion matrices, classification reports, and a model comparison chart.

### Step 6: Launch the Web App
```bash
streamlit run app.py
```
Opens an interactive web app in your browser!

---

## 🧪 Key Design Decisions & Justifications

### Why arXiv API instead of a pre-made Kaggle dataset?
- Demonstrates real-world **data engineering** skill
- Abstracts are messy (LaTeX, special characters), requiring actual NLP cleaning
- Makes the project reproducible and updatable at any time

### Why TF-IDF as the baseline?
- TF-IDF captures **domain-specific vocabulary patterns** effectively
- It's fast, interpretable, and very competitive on this task
- Every proper ML project needs a baseline to compare advanced models against

### Why DistilBERT over full BERT?
- DistilBERT is **40% smaller and 60% faster** than BERT
- It retains **97% of BERT's performance** (Sanh et al., 2019)
- A practical engineering decision: choosing the right model for your hardware constraints

### Why Logistic Regression over Naive Bayes?
- Naive Bayes assumes word independence (clearly wrong for language)
- Logistic Regression makes no such assumption and typically achieves **5–10% higher accuracy** on text classification
- We still train Naive Bayes for comparison to demonstrate model awareness

---

## 📊 Results

| Model | Test Accuracy |
|:---|:---:|
| TF-IDF + Logistic Regression (Baseline) | ~88% |
| DistilBERT (Fine-tuned) | ~94% |

---

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| Language | Python 3.11 |
| NLP Preprocessing | NLTK |
| Feature Extraction | Scikit-learn (TF-IDF) |
| ML Models | Scikit-learn (Logistic Regression, Naive Bayes) |
| Deep Learning | PyTorch + Hugging Face Transformers |
| Model: DistilBERT | `distilbert-base-uncased` (Hugging Face) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit |
| Data Source | arXiv API |

---

## 👤 Author

**[Your Name]** | 3rd Year B.Tech Student  
*Built for the ML/NLP Engineer portfolio.*  
[LinkedIn](https://linkedin.com) | [GitHub](https://github.com)
