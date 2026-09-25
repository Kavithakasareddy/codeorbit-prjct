# Fake News Detection Using Machine Learning

## 📌 Project Overview

Fake News Detection is a machine learning project designed to classify news articles as **Fake** or **Real**.

The project uses Natural Language Processing (NLP) techniques to process news text and machine learning to identify patterns in the data.

## 🎯 Objectives

- Detect whether a news article is Fake or Real.
- Process news text using Natural Language Processing.
- Train a machine learning classification model.
- Provide predictions based on the trained model.
- Develop a simple and easy-to-use application.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Machine Learning

## 📊 Dataset

The project contains two main datasets:

- `fake.csv` – contains fake news articles.
- `true.csv` – contains real news articles.

The original datasets contain:

- **23,481 Fake news records**
- **21,417 Real news records**

Because the original datasets are large, smaller sample datasets were created for the GitHub repository:

- `fake_sample.csv` – 5,000 randomly selected fake news records.
- `true_sample.csv` – 5,000 randomly selected real news records.

The original datasets are kept locally and are not uploaded to GitHub because of their large file size.

## 📂 Project Structure

```text
codeorbit-prjct/
│
├── app.py
├── README.md
│
└── dataset/
    ├── fake_sample.csv
    └── true_sample.csv
