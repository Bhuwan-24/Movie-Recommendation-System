# Amazon Product Recommendation System

click to view: https://huggingface.co/spaces/Bhuwan-24/E-commerce_Recommender
 
A content-based product recommendation system built on 117k+ Amazon products. This was a personal project to learn how real recommendation systems work from messy raw data all the way to a deployed web app.
 

 
## What this project does
 
Given a product title and category, it finds the most similar products from a dataset of 117,000+ Amazon listings and returns them ranked by similarity score.
 
The dataset had 24,805 products with no category label. Instead of just dropping them, I trained a text classifier to predict the missing categories which was honestly the more interesting part of the project.
 

 
## Project structure
 
```
├── categories_prediction.ipynb   # trains classifier to fill missing categories
├── model.ipynb                   # builds the TF-IDF matrix and saves artifacts
├── text_clean.py                 # text preprocessing shared across both notebooks
├── recommend.py                  # loads saved artifacts and handles recommendations
├── app.py                        # Streamlit frontend
└── requirements.txt
```
 

 
## How I built it
 
### Part 1 — Filling the missing categories
 
24,805 out of 117,243 products had no `main_category` label. I trained three classifiers on the labeled portion and compared them:
 
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 83% | 0.68 | 0.84 |
| Multinomial Naive Bayes | 78% | 0.38 | 0.76 |
| **LinearSVC** | **88%** | **0.74** | **0.88** |
 
LinearSVC won by a clear margin so I used it to predict categories for all the unlabeled rows. All three models use a TF-IDF pipeline with bigrams and `class_weight="balanced"` since the categories are pretty imbalanced (Amazon Fashion alone has 36k products).
 
### Part 2 — Building the recommender
 
Instead of treating the product as a single blob of text, I vectorized title, category, and description separately and gave each field a different weight before combining them:
 
```
title weight:       3.0  — most specific, strongest signal
category weight:    2.0  — keeps results in the right domain  
description weight: 1.0  — extra context
```
 
The three weighted sparse matrices get stacked horizontally into one big product matrix. At query time, I transform the input the same way and compute cosine similarity against the full matrix to get ranked results.
 
The vectorizers and product matrix are saved with `joblib` so the app loads them once at startup instead of rebuilding everything each time.
 
### Part 3 — The Streamlit app
 
Pretty straightforward text input for the query, optional category field, a slider for how many results you want, and a table showing the results with scores.
 

 
## Running it locally
 
```bash
pip install -r requirements.txt
```
 
Run the notebooks in order first to generate the `.pkl` files:
 
```bash
jupyter notebook categories_prediction.ipynb
jupyter notebook model.ipynb
```
 
Then launch the app:
 
```bash
streamlit run app.py
```
 

 
## Example
 
Query: `"Nike shoes"` in category `"fashion"`
 
| title | main_category | score |
|---|---|---|
| nike girls boyshorts | amazon fashion | 0.847 |
| nike casual | amazon fashion | 0.831 |
| nike modern | amazon fashion | 0.819 |
| nike hoodies | amazon fashion | 0.804 |
| nike unisex-child modern | amazon fashion | 0.798 |
 

 
## What I'd improve with more time
 
- Right now cosine similarity runs over the full 117k product matrix on every query which is fine for a demo but wouldn't scale. FAISS or a similar ANN library would fix that.
- TF-IDF only matches on exact words — "running shoes" and "jogging footwear" wouldn't match well. A sentence transformer model would handle that much better.
- No proper evaluation on the recommender side. I'd want to add precision@k at minimum.

 
## Stack
 
Python · scikit-learn · NLTK · scipy · pandas · joblib · Streamlit