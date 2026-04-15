# Kannada-English Humor Detection

A comparative NLP study that fine-tunes and evaluates **5 state-of-the-art transformer models** on a **Kannada-English code-mixed humor classification** task. Each notebook implements the full pipeline: preprocessing, language detection, translation, tokenization, training, and evaluation.

---

## Overview

This project classifies text as **Humor** or **Non-Humor** using a dataset of Kannada-English jokes (`df.csv`). The key challenge is handling **code-mixed text**, sentences that blend Kannada script and English words, which requires language detection and translation as preprocessing steps before feeding text into transformer models.

Five transformer architectures are benchmarked against each other on the same dataset and task.

---

## Repository Structure

```
NLP_models/
├── BERT_NLP_with_preprocessing_clean.ipynb   # mBERT fine-tuning pipeline
├── DISTILBERT.ipynb                           # DistilBERT fine-tuning pipeline
├── ELECTRA_clean.ipynb                        # ELECTRA fine-tuning pipeline
├── INDICBERT_clean.ipynb                      # IndicBERT fine-tuning pipeline
├── ROBERTA_clean.ipynb                        # RoBERTa fine-tuning pipeline
```

---

## The Code-Mixed Challenge

The input data mixes **Kannada script** (Unicode range `0x0C80–0x0CFF`) and **English words** in the same sentence. The preprocessing pipeline handles this in two stages:

1. **Special character removal** - strips punctuation and noise tokens
2. **Language detection & translation** - each word is classified as Kannada or English using Unicode analysis; English words are translated to Kannada via Google Translate (`deep_translator`) to produce a fully Kannada-normalized input before tokenization

---

## Models Compared

| Notebook | Model | Key Characteristic |
|---|---|---|
| `BERT_NLP_with_preprocessing_clean.ipynb` | `bert-base-multilingual-cased` | 104-language multilingual BERT |
| `DISTILBERT.ipynb` | DistilBERT | Lighter, faster distilled version of BERT |
| `ELECTRA_clean.ipynb` | ELECTRA | Replaced Token Detection pre-training objective |
| `INDICBERT_clean.ipynb` | IndicBERT | Trained specifically on Indic languages incl. Kannada |
| `ROBERTA_clean.ipynb` | RoBERTa | Robustly optimized BERT with dynamic masking |

---

## Pipeline (per notebook)

All 5 notebooks follow the same structure:

### 1. Install Dependencies
```bash
pip install indic-transliteration langdetect deep_translator transformers torch
```

### 2. Preprocessing
- Load `df.csv` (columns: `JOKES`, `LABEL`)
- Remove special characters
- Detect language of each word (Kannada vs. English via Unicode)
- Translate English words → Kannada using `GoogleTranslator`

### 3. Tokenization
- Tokenize with the model's corresponding HuggingFace tokenizer
- `max_length=64`, padding, truncation, attention masks
- Output: `TensorDataset` of input IDs, attention masks, labels

### 4. Train/Val/Test Split
- 80% train / 10% validation / 10% test
- `DataLoader` with `RandomSampler` (train) and `SequentialSampler` (val/test)

### 5. Training
- **Optimizer:** AdamW (`lr=5e-5`, `eps=1e-8`)
- **Scheduler:** Linear warmup
- **Epochs:** 4
- **Batch size:** 32
- Gradient clipping (`max_norm=1.0`)

### 6. Evaluation
- Loss and accuracy reported per epoch
- Final test set evaluation
- Confusion matrix plotted with seaborn
---

## Tech Stack

- **Python 3.9**
- **PyTorch** — model training & inference
- **HuggingFace Transformers** — pre-trained transformer models & tokenizers
- **TensorFlow / Keras** — additional DL utilities
- **scikit-learn** — train/test split, confusion matrix, classification report
- **deep_translator** — Google Translate API wrapper for Kannada translation
- **langdetect** — language detection
- **indic-transliteration** — Indic script handling
- **NLTK** — text preprocessing
- **Pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — confusion matrix visualization

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/madhulikavikraman/NLP_models.git
cd NLP_models
```

### 2. Install Dependencies

```bash
pip install indic-transliteration langdetect deep_translator transformers torch \
            tensorflow scikit-learn pandas numpy matplotlib seaborn nltk jupyter
```

### 3. Add the Dataset

Create a dataset called `df.csv` and place it in the root directory. It should have:
- `JOKES` — the raw text (Kannada-English code-mixed)
- `LABEL` — binary label (`0` = Non-Humor, `1` = Humor)

### 4. Run a Notebook

```bash
jupyter notebook BERT_NLP_with_preprocessing_clean.ipynb
```

> **GPU recommended.** The notebooks automatically detect and use CUDA if available:
> ```python
> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
> ```
> Running on Google Colab with a GPU runtime is the easiest way to reproduce results.

---

## License

This project is open source and available for academic and research use.
