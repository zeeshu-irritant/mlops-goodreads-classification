# MLOps Pipeline: Goodreads Genre Classification

## Project Overview
This repository contains an end-to-end MLOps pipeline for fine-tuning a Large Language Model (LLM) on text classification. A pre-trained `distilbert-base-cased` model was fine-tuned on the UCSD Goodreads Reviews dataset to classify book reviews into 8 distinct genres. The project demonstrates cloud-based training, secure secrets management, experiment tracking, and model deployment.

## Setup Instructions
To replicate this environment or run the inference pipeline locally, install the dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

To run inference using the deployed model via the Hugging Face `pipeline`:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="zeeshan-hf/distilbert-goodreads-genres")
prediction = classifier("The magic system and world-building were absolutely phenomenal!")
print(prediction)
```

## Training Platform
The model was trained using a **Kaggle Notebook** leveraging dual NVIDIA T4 GPUs. All API keys (Weights & Biases, Hugging Face) were securely managed using Kaggle Secrets to prevent credential leakage.

## Results
Below are the final evaluation metrics captured on the held-out test dataset:

| **Metric** | **Score** |
|---|---|
| **Accuracy** | 0.546 |
| **Weighted F1 Score** | 0.555 |
| **Evaluation Loss** | 2.481 |

## Project Links
* **Kaggle Notebook:** [Kaggle Notebook | MLOps: Assignment 2 - Fine Tuning Classification](https://www.kaggle.com/code/zeeshang25ait2135/mlops-assignment-2-fine-tuning-classification)
* **Hugging Face Model:** [Hugging Face | MLOps: Assignment 2 - DistilBERT for Goodreads Genres](https://huggingface.co/zeeshan-hf/distilbert-goodreads-genres)
* **Weights & Biases Dashboard:** [W&B Project | MLOps: Assignment 2 - DistilBERT Fine Tuning](https://wandb.ai/zeeshu-irritant-prom-iit-rajasthan/mlops-assignment2)
