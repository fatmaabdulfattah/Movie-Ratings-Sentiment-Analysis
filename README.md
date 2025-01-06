# Movie-Ratings-Sentiment-Analysis

This repository contains a **Kaggle notebook** and a **Streamlit app** for performing sentiment analysis on movie reviews. The project uses a Bidirectional LSTM model trained on the IMDB movie review dataset to classify reviews as either positive or negative.

---

## Overview

This project demonstrates a complete workflow for sentiment analysis:
1. **Kaggle Notebook**: A notebook that preprocesses the IMDB movie review dataset, trains a Bidirectional LSTM model, and evaluates its performance.
2. **Streamlit App**: A web-based interface that allows users to input a movie review and get a sentiment prediction (positive or negative) along with the model's confidence score.

---

## Kaggle Notebook

### Description
The Kaggle notebook performs sentiment analysis on the IMDB movie review dataset. It includes the following steps:
- **Data Loading**: Loads the dataset containing 40,000 movie reviews.
- **Data Cleaning and Preprocessing**: Duplicates are removed and preprocesses the text data (lowercasing, removing punctuation, stopwords, etc.).
- **Tokenization**: Text data is converted into sequences using the `Tokenizer` class.
- **Model Building**: A Bidirectional LSTM model is built using `tensorflow.keras`.
- **Model Training**: The model is trained for 5 epochs with a validation split of 0.2.
- **Evaluation**: Evaluates the model using accuracy on the test set, and a confusion matrix is plotted.
- **Prediction**: Demonstrates how to predict the sentiment of new reviews.


### How to Use
1. Open the notebook on Kaggle: [[movie rating sentiment analysis](https://www.kaggle.com/code/fatmaabdulfattah/movie-ratings-sentiment-analysis)](#) 
2. Click "Copy and Edit" to create your own version.
3. Run the cells sequentially to reproduce the analysis or modify the code.



## Streamlit App

### Description
The Streamlit app provides a user-friendly interface for predicting the sentiment of movie reviews. Users can input a review, and the app will classify it as positive or negative using the trained LSTM model.

### How to Run Locally





### Dependencies
The notebook requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `scikit-learn`
- `nltk`
- `streamlit`
- `pickle`

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn nltk streamlit

Clone this repository:
```bash
git clone https://github.com/your-username/your-repo-name.git

Navigate to the Streamlit app directory:
```bash
cd your-repo-name/streamlit-app

Install the required dependencies:
```bash
pip install -r requirements.txt

Run the Streamlit app:
```bash
streamlit run app.py
