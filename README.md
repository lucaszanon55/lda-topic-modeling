# LDA Topic Modeling Pipeline

This repository contains a Python pipeline for performing topic modeling on PDF documents using Latent Dirichlet Allocation (LDA).

## Features
- Extracts text from PDFs
- Cleans and preprocesses text (stopwords removal, lemmatization)
- Builds a document–term matrix
- Finds the optimal number of topics using coherence scores
- Trains an LDA model (reproducible with fixed random seed)
- Generates an interactive visualization with pyLDAvis

## How to Use
1. Put your PDF file in the `data/` folder.
2. Edit `topic_modeling_pipeline.py` and change the variable:
   ```python
   pdf_file_path = 'data/your_corpus.pdf'
