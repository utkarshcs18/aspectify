<div align="center">

# ⚡ Aspectify
### Fine-Grained Aspect-Based Sentiment Analysis (ABSA) for Hinglish & English E-Commerce Reviews

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br/>

> **"Battery bahut accha hai but delivery 10 din late aayi aur packaging bekar thi."**  
> Traditional sentiment analysis gives this review a flattened `Neutral` or `Positive` score.  
> **Aspectify** disentangles mixed opinions into actionable feature-level intelligence.

<br/>
</div>

---

## 📌 Executive Summary

When customers review products on Indian e-commerce platforms (Flipkart, Amazon India, Meesho), reviews rarely express a single sentiment. Instead, they code-mix **Hindi and English (Hinglish)**, praising one attribute while strongly criticizing another:

$$\text{Review} \longrightarrow \begin{cases} \mathbf{Battery:} & \text{Positive} \quad \small\text{(bahut accha)} \\ \mathbf{Delivery:} & \text{Negative} \quad \small\text{(10 din late aayi)} \\ \mathbf{Packaging:} & \text{Negative} \quad \small\text{(bekar thi)} \end{cases}$$

Standard polarity classifiers mask critical product defects. **Aspectify** tackles this problem via a two-stage classical machine learning pipeline equipped with phonetic/character $n$-gram engineering, Hinglish phonetic normalizers, negation preservation, and clause boundary splitting.

---

## 🎯 The Core Problem & Challenges

| Challenge | Why Standard NLP Fails | Aspectify's Solution |
| :--- | :--- | :--- |
| **Sentiment Flattening** | *"Good camera but dead battery"* gets compressed into an unhelpful neutral label. | Deconstructs text into individual aspect-sentiment tuples. |
| **Code-Mixed Hinglish** | Hindi written phonetically in Latin script (*"paisa vasool"*, *"bakwas"*, *"mast"*). | Curated Hinglish lexical dictionaries + Romanized sentiment mappings. |
| **Noisy Spelling Variants** | Same word spelled 5+ ways: `accha`, `acha`, `achha`, `achchha`, `axha`. | Sub-word **Character $n$-grams** (`analyzer="char_wb"`, $n \in [2, 5]$) + canonical mapping. |
| **Negation Inversion** | Aggressive stopword removal strips `"nahi"` or `"not"`, converting *"accha nahi"* $\to$ *"accha"* (+ve). | Context-aware stopword filtering retaining polarity inverters (`nahi`, `nhi`, `mat`, `never`). |
| **Label Scarcity** | E-commerce sets only offer star ratings, not multi-aspect ground truths. | **Weak supervision** (lexicon bootstrapping) + rigorously verified **Human Gold Test Set (500+ reviews)**. |

---

## ✨ Key Features

- 🔍 **Two-Stage Modular Architecture:** Separates aspect identification (multi-label) from sentiment polarity (multi-class).
- 🇮🇳 **First-Class Hinglish Support:** Native handling of Romanized Hindi idioms, slang, and colloquial expressions.
- 🔠 **Subword Character $n$-gram Embeddings:** Resilient to phonetic typos, slang abbreviations, and non-standard transliterations.
- ✂️ **Acoustic / Syntactic Clause Splitting:** Uses discourse markers (`but`, `lekin`, `par`, `aur`, `,`) and window-based attention zones around aspect anchors.
- 🏷️ **Weak Supervision + Gold Test Benchmark:** Bootstrapped using domain lexicons, rigorously evaluated on hand-annotated ground truth.
- 📊 **Interactive Streamlit Web Dashboard:** Single-review inference with colored aspect badges + bulk CSV analytics with aspect breakdown graphs.

---

## 🧠 System Architecture

```
                                      ┌──────────────────────────────────────────────┐
                                      │           Incoming Customer Review           │
                                      │ "Battery accha hai but delivery bekar thi"  │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                             ▼
                                     ┌────────────────────────────────────────────────┐
                                     │         Preprocessing & Normalization          │
                                     │ • Lowercasing & emoji-to-sentiment translation │
                                     │ • Hinglish spelling canonicalization           │
                                     │ • Negation preservation (nahi, mat, not)       │
                                     └───────────────────────┬────────────────────────┘
                                                             │
                                                             ▼
                                     ┌────────────────────────────────────────────────┐
                                     │         Feature Extraction (TF-IDF)            │
                                     │ • Word n-grams: (1, 2)                         │
                                     │ • Char-WB n-grams: (2, 5)                      │
                                     └───────────────────────┬────────────────────────┘
                                                             │
                                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Aspect Detection (Multi-Label Classification)                                                                                │
│  One-vs-Rest Classifiers across: [Battery, Delivery, Price, Quality, Packaging, Customer Service]                                      │
└────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┘
                                                             │
                                          Detected: [Battery, Delivery]
                                                             │
                                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Clause Segmentation & Sentiment Classification                                                                               │
│  • Syntactic splitting on conjunctions ("but", "lekin", "par", ",")                                                                    │
│  • Aspect window extraction:                                                                                                           │
│    - Clause 1: "battery accha hai"      ──► Classifier ──► 🟢 Positive                                                                │
│    - Clause 2: "delivery bekar thi"     ──► Classifier ──► 🔴 Negative                                                                │
└────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┘
                                                             │
                                                             ▼
                                     ┌────────────────────────────────────────────────┐
                                     │           Structured Prediction Output         │
                                     │    { "battery": "positive",                    │
                                     │      "delivery": "negative" }                  │
                                     └────────────────────────────────────────────────┘
```

---

## 🏷️ Aspect Ontology & Lexicon Design

The system tracks **6 high-impact e-commerce dimensions**:

| Aspect | Core English Keywords | Hinglish / Romanized Hindi Expressions |
| :--- | :--- | :--- |
| 🔋 **Battery** | `battery`, `charging`, `backup`, `drain`, `mah`, `charger` | `charge nahi tikta`, `battery backup mast`, `jaldi khatam` |
| 🚚 **Delivery** | `delivery`, `courier`, `shipping`, `dispatch`, `late`, `delay` | `bahut late aaya`, `time pe nahi mila`, `courier bekar` |
| 💰 **Price** | `price`, `cost`, `expensive`, `cheap`, `worth`, `affordable` | `paisa vasool`, `mehenga`, `loot liya`, `sasta quality` |
| ⭐ **Quality** | `quality`, `material`, `durability`, `build`, `performance` | `ekdum solid`, `ghatiya material`, `kharab ho gaya` |
| 📦 **Packaging** | `packaging`, `box`, `bubble wrap`, `damaged`, `seal`, `broken` | `package fata hua tha`, `dabba toota`, `seal open` |
| 🎧 **Service** | `support`, `replacement`, `warranty`, `customer care`, `refund` | `koi sunwayi nahi`, `refund nahi mila`, `service center bekar` |

---

## 📊 Results & Evaluation

> [!NOTE]  
> Models were evaluated on our **500+ Hand-Labeled Gold Test Set** using stratified splits to prevent data contamination from weak-supervision heuristics.

### 1. Aspect Detection (Multi-Label Macro-Averaged)

| Model Architecture | Feature Representation | Precision | Recall | Macro F1 |
| :--- | :--- | :---: | :---: | :---: |
| Naive Bayes (Multinomial) | Word TF-IDF $(1, 2)$ | 0.72 | 0.65 | 0.68 |
| Logistic Regression (OvR) | Word TF-IDF $(1, 2)$ | 0.81 | 0.76 | 0.78 |
| Logistic Regression (OvR) | Word $(1, 2)$ + Char-WB $(2, 5)$ | 0.86 | 0.83 | **0.84** |
| **Linear SVM (OvR) 🏆** | **Word $(1, 2)$ + Char-WB $(2, 5)$** | **0.88** | **0.85** | **0.86** |

### 2. Aspect Sentiment Polarity (Clause-Level 3-Class)

| Model Architecture | Features | Positive F1 | Neutral F1 | Negative F1 | Macro F1 | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Bayes | Word TF-IDF | 0.74 | 0.49 | 0.72 | 0.65 | 71.4% |
| Logistic Regression | Word + Char-WB | 0.83 | 0.62 | 0.82 | 0.76 | 80.2% |
| **Linear SVM 🏆** | **Word + Char-WB** | **0.86** | **0.67** | **0.85** | **0.79** | **83.1%** |

> **Key Empirical Finding:** Adding character-level $n$-grams (`char_wb`, $2$ to $5$) resulted in a **+6.0% macro F1 jump** on Hinglish text, directly mitigating non-standard spelling permutations (`bakwas` vs `bkwas` vs `bakwaas`).

---

## 🔬 In-Depth Error Analysis

> *A model is only as good as our understanding of where it fails.*

Here is a systematic taxonomy of the primary failure modes discovered during qualitative audits:

```
                                   Error Taxonomy
                                         │
        ┌───────────────────┬────────────┴───────┬──────────────────┐
        ▼                   ▼                    ▼                  ▼
    🎭 Sarcasm          ⚡ Negation         🔀 Mixed Clause    🕵️ Implicit
   & Irony Blindness   Displacement        Interference        Aspect Mentions
```

### 1. Sarcasm & Irony
* **Example:** *"Wah bhai wah, 2 din mein phone dead ho gaya. Great product!"*
* **Ground Truth:** `Quality -> Negative` | **Model Prediction:** `Quality -> Positive`
* **Root Cause:** TF-IDF assigns heavy positive weights to *"Wah"*, *"Great"*, and *"product"* without contextual phrase comprehension.
* **Remediation Strategy:** Requires bidirectional attention mechanisms (e.g., fine-tuned `IndicBERT` or `HingBERT`).

### 2. Complex Negation Displacement
* **Example:** *"Aisa nahi hai ki battery kharab hai, par charging slow hai."*
* **Ground Truth:** `Battery -> Neutral/Positive`, `Battery -> Negative (charging)` | **Model Prediction:** `Battery -> Negative`
* **Root Cause:** Proximity of *"nahi"* + *"kharab"* triggered negative polarity despite double negation.
* **Remediation Strategy:** Dependency parsing for Hinglish or increasing window boundaries with syntactic dependency tags.

### 3. Mixed Sentiment in a Single Clause
* **Example:** *"Battery life thik thak hai par heating issue bohot zyada hai."*
* **Ground Truth:** `Battery -> Neutral`, `Quality/Battery -> Negative` | **Model Prediction:** Flips both to negative.
* **Root Cause:** Overlap in aspect token windows when conjunction boundaries are omitted or ambiguous.
* **Remediation Strategy:** Sentence constituency boundary detection tailored to Indic discourse connectives (`halanki`, `bawajood`, `magar`).

### 4. Implicit Aspect Mentions
* **Example:** *"Dabba khola toh screen pe crack tha."*
* **Ground Truth:** `Packaging/Delivery -> Negative` | **Model Prediction:** No aspect detected.
* **Root Cause:** No explicit keyword matched `packaging` or `delivery` in the surface lexicon.
* **Remediation Strategy:** Dense semantic vector embeddings (Sentence-Transformers / `bge-m3`) rather than sparse lexical TF-IDF.

---

## 💻 Tech Stack

<div align="left">

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Core Language** | Python 3.9+ |
| **Data & Vectorization** | NumPy, Pandas, Scikit-Learn (TF-IDF, OneVsRestClassifier) |
| **NLP & Text Processing** | NLTK, Regex, Emoji, Custom Hinglish Normalizer |
| **Visualization & Metrics** | Matplotlib, Seaborn, Scikit-Learn Metrics |
| **Model Persistence** | Joblib |
| **Web UI & Deployment** | Streamlit, Hugging Face Spaces |

</div>

---

## 📂 Project Organization

```bash
aspectify/
├── data/
│   ├── raw/                   # Raw e-commerce dumps (Flipkart, Amazon India, SemEval)
│   ├── processed/             # Cleaned, tokenized, and normalized reviews
├── notebooks/
│   ├── 01_data_collection_eda.ipynb    # Review lengths, rating distribution, language mix
│   ├── 02_preprocessing_pipeline.ipynb # Hinglish normalizers, negation preservation
│   ├── 03_weak_labeling_gold_set.ipynb # Lexicon rule-engine & human annotation verification
│   ├── 04_aspect_detection_models.ipynb# Multi-label OvR (Logistic Regression, SVM, NB)
│   ├── 05_aspect_sentiment_models.ipynb# Clause segmentation & sentiment classification
│   └── 06_error_analysis_audit.ipynb   # Qualitative breakdown of failure modes
├── src/
│   ├── preprocessing.py       # Cleaning, emoji mapping, spelling normalizer
│   ├── features.py            # Word & character n-gram TF-IDF pipelines
│   ├── train.py               # Model training and hyperparameter tuning routines
│   ├── predict.py             # End-to-end inference API
│   └── evaluation.py          # Metric calculations, confusion matrices, classification reports
├── models/                    # Serialized joblib models and vectorizers
├── app/
│   └── app.py                 # Interactive Streamlit application
├── tests/                     # Unit tests for preprocessing & tokenizer sanity
├── requirements.txt           # Environment dependencies
└── README.md
```

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/utkarshcs18/aspectify.git
cd aspectify
```

### 2. Set Up a Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate on Linux/macOS:
source venv/bin/activate

# Activate on Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application Locally
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Quick Python Inference Example

```python
from src.predict import AspectSentimentPipeline

# Initialize end-to-end inference engine
pipeline = AspectSentimentPipeline.load(model_dir="models/")

review = "Camera ekdum mast hai but battery 2 ghante mein drain ho gayi aur packaging kharab thi!"
results = pipeline.predict(review)

for aspect, sentiment in results.items():
    print(f"[{aspect.upper()}]: {sentiment}")
```

**Output:**
```text
[CAMERA]: positive
[BATTERY]: negative
[PACKAGING]: negative
```

---

## 🚀 Live Demo & Preview

Try out the interactive demo hosted on **Hugging Face Spaces**:  
👉 **[Launch Aspectify on Hugging Face Spaces](https://huggingface.co/spaces)**

```
┌────────────────────────────────────────────────────────────────────────┐
│  ⚡ Aspectify - Review Intelligence Dashboard                          │
│                                                                        │
│  Input Review:                                                         │
│  [ Battery bahut accha hai but delivery 10 din late thi...          ]  │
│                                                                        │
│  Detected Aspects & Polarity:                                          │
│  ┌──────────────────────┐  ┌──────────────────────┐                    │
│  │  🔋 BATTERY: Positive│  │  🚚 DELIVERY: Negative│                  │
│  └──────────────────────┘  └──────────────────────┘                    │
│                                                                        │
│  📊 Batch CSV Analytics:                                               │
│  [ Upload reviews.csv ] -> 📈 Aspect Sentiment Distribution Chart      │
└────────────────────────────────────────────────────────────────────────┘
```


---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---
