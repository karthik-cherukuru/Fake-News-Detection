# Fake News Detection using Machine Learning and Deep Learning

This project focuses on detecting fake news articles using various supervised learning models, including:

- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Feed Forward Neural Network (TensorFlow)

## Features
- Text preprocessing: tokenization, lemmatization, stopword removal
- TF-IDF vectorization of combined title + text
- Model evaluation using:
  - Accuracy
  - Classification report
  - Confusion matrix (Seaborn heatmaps)
  - Training & prediction time analysis

## Dataset
The dataset used contains `title`, `text`, and `label` columns to distinguish between real and fake news articles.

Link to dataset:  
[Detecting Fake News Dataset on Kaggle](https://www.kaggle.com/datasets/amirmotefaker/detecting-fake-news-dataset/data)

> Ensure the file is named `news.csv` and placed in the same directory as your notebook.

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- seaborn
- matplotlib
- tensorflow (for neural network)


## Author
Karthik Cherukuru
