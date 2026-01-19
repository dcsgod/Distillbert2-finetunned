# Distillbert2-finetunned
---
library_name: transformers
tags:
- movie
- question-answering
- text-generation
license: apache-2.0
datasets:
- HiTruong/movie_QA
- Pablinho/movies-dataset
language:
- en
metrics:
- bertscore
base_model:
- distilbert/distilgpt2
---

# DistilGPT2 Movie Question Answering Model

## Overview

This repository contains a **fine-tuned DistilGPT2 model** designed to answer **movie-related questions**.  
The model can respond to queries about movie plots, directors, cast, and general trivia using a simple prompt format:


The goal of this project is to demonstrate **domain-specific fine-tuning** of a lightweight language model using publicly available movie datasets.

---

## Model Details

- **Model type:** Causal Language Model (Text Generation)
- **Base model:** distilbert/distilgpt2
- **Language:** English
- **License:** Apache 2.0
- **Fine-tuning objective:** Movie Question Answering

### Author

- **Name:** Ravi Kumar  
- **Hugging Face:** https://huggingface.co/iravikr  
- **GitHub:** https://github.com/dcsgod  
- **LinkedIn:** https://linkedin.com/in/ravi3kr  
- **Email:** rk9128557489@gmail.com  

---

## Training Data

The model was fine-tuned using the following datasets from Hugging Face:

1. **HiTruong/movie_QA**  
   - Contains curated movie-related question–answer pairs.
   - Used as direct supervised QA training data.

2. **Pablinho/movies-dataset**  
   - Contains movie metadata such as plot summaries and descriptions.
   - Converted into question–answer format during preprocessing.

All training samples were normalized into the following text format:


---

## Training Procedure

### Preprocessing

- Converted structured metadata into natural language Q&A pairs.
- Tokenized using the DistilGPT2 tokenizer.
- Maximum sequence length: 256 tokens.
- Padding token set to EOS token.

### Hyperparameters

- **Training regime:** fp16 mixed precision
- **Optimizer:** AdamW
- **Learning rate:** 5e-5
- **Epochs:** 3
- **Effective batch size:** 16 (via gradient accumulation)
- **Training steps:** 13,095

### Compute

- **Hardware:** NVIDIA T4 GPU
- **Platform:** Google Colab
- **Training time:** ~1.1 hours

### Training Result

- **Final training loss:** ~2.29  
- Loss steadily decreased, indicating stable learning without collapse or overfitting.

---

## Intended Use

### Direct Use

- Movie question answering chatbots
- Movie trivia applications
- Educational demos
- Lightweight domain-specific assistants

### Downstream Use

- Can be combined with retrieval systems (FAISS / Chroma) for RAG-based movie QA
- Can be deployed via FastAPI or Streamlit
- Suitable for experimentation and learning purposes

### Out-of-Scope Use

- Real-time or up-to-date movie facts without retrieval
- Complex multi-hop reasoning
- High-stakes or authoritative factual systems
- Long conversational memory use cases

---

## Limitations and Risks

- The model may hallucinate facts for movies not present in the training data.
- Knowledge is limited to the datasets used during fine-tuning.
- Bias may exist toward popular or English-language films.

### Recommendations

- Use retrieval augmentation for production systems.
- Validate responses when factual correctness is critical.
- Avoid using the model as a sole source of truth.

---

## How to Use

### Loading the Model

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="iravikr/distilgpt2-movieqa",
    tokenizer="iravikr/distilgpt2-movieqa"
)

pipe(
    "Question: Who directed Inception?\nAnswer:",
    max_new_tokens=40,
    temperature=0.7,
    top_p=0.9
)
Question: Who directed Inception?
Answer: Inception was directed by Christopher Nolan.

---

## Training Procedure

### Preprocessing

- Converted structured movie metadata into natural language QA pairs.
- Tokenized using the DistilGPT2 tokenizer.
- Maximum sequence length: 256 tokens.
- Padding token set to EOS token.

### Hyperparameters

- **Training regime:** fp16 mixed precision
- **Optimizer:** AdamW
- **Learning rate:** 5e-5
- **Epochs:** 3
- **Effective batch size:** 16 (via gradient accumulation)
- **Total training steps:** 13,095

---

## Hardware and Compute

- **GPU:** NVIDIA T4 (16 GB VRAM)
- **CPU:** Intel Xeon (Google Colab backend)
- **RAM:** ~12 GB
- **Platform:** Google Colab
- **Training time:** Approximately 1.1 hours

---

## Training Outcome

- **Initial training loss:** ~2.98  
- **Final training loss:** ~2.29  

The steady loss reduction indicates stable convergence without overfitting.

---

## Intended Use

### Direct Use

- Movie question answering systems
- Movie trivia bots
- Educational and demonstration projects
- Lightweight domain-specific assistants

### Downstream Use

- Integration with retrieval systems (FAISS / Chroma) for RAG-based QA
- Deployment via FastAPI or Streamlit
- Extension for recommendation-style prompts

### Out-of-Scope Use

- Real-time or up-to-date movie facts without retrieval
- Complex multi-hop reasoning
- High-stakes decision-making systems
- Long-context conversational memory

---

## Limitations and Risks

- The model may hallucinate facts for movies not seen during training.
- Knowledge is limited to the scope of the training datasets.
- Dataset bias toward popular and English-language films may exist.

### Recommendations

- Use retrieval augmentation for production deployments.
- Validate responses for factual accuracy.
- Avoid use as a sole authoritative source.

---

## How to Use

### Loading the Model

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="iravikr/distilgpt2-movieqa",
    tokenizer="iravikr/distilgpt2-movieqa"
)

pipe(
    "Question: Who directed Inception?\nAnswer:",
    max_new_tokens=40,
    temperature=0.7,
    top_p=0.9
)
