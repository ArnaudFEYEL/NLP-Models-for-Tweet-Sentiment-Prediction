# NLP Models for Tweet Sentiment Prediction

This repository contains multiple models and approaches for analyzing tweet sentiments. The models work with a dataset of airline-related tweets, accessible [here](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

## Models Included
1. **Lexicon-Based Analysis**:
   - `lexicon_main.py` uses rule-based heuristics (VADER model) to assign sentiment based on predefined word scores.

2. **CBOW Neural Network**:
   - `CBOW_NN_model.py` predicts word sentiments by analyzing their surrounding context using a shallow neural network.

3. **LSTM Neural Network**:
   - `LSTM_NN_model.py` captures sequential dependencies in tweets, ideal for complex sentence structures.

## Usage Workflow
1. Install dependencies using `check_packages.py`.
2. Preprocess sentences with `easy_sentence_maker.py`.
3. Run:
   - `lexicon_main.py` for rule-based analysis.
   - `CBOW_NN_model.py` or `LSTM_NN_model.py` for neural methods.



