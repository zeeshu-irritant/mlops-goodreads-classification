# MLOps Pipeline: Goodreads Genre Classification

## Project Overview
This repository contains an end-to-end MLOps pipeline for fine-tuning a Large Language Model (LLM) on text classification. A pre-trained `distilbert-base-cased` model was fine-tuned on the UCSD Goodreads Reviews dataset to classify book reviews into 8 distinct genres. The project demonstrates cloud-based training, secure secrets management, experiment tracking, and model deployment.

## Setup Instructions
To replicate this environment or run the inference pipeline locally, install the dependencies using the provided requirements file:

```bash
pip install -r requirements.txt

from transformers import pipeline

classifier = pipeline("text-classification", model="zeeshan-hf/distilbert-goodreads-genres")
prediction = classifier("The magic system and world-building were absolutely phenomenal!")
print(prediction)
