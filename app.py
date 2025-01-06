import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
@st.cache_resource
def load_lstm_model():
    model = load_model(r"D:\movie_rating_project\lstm_sentiment_model.h5")  
    return model

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open(r"D:\movie_rating_project\tokenizer.pkl", 'rb') as f:  
        tokenizer = pickle.load(f)
    return tokenizer

# Preprocess the review text
def preprocess_review(review, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence

# Predict the sentiment
def predict_sentiment(review, model, tokenizer, max_length):
    processed_review = preprocess_review(review, tokenizer, max_length)
    prediction = model.predict(processed_review)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return sentiment, confidence

# Streamlit UI
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment.")

# User input
review = st.text_area("Enter your movie review:")

# Predict sentiment
if st.button("Analyze Sentiment"):
    if review.strip():
        lstm_model = load_lstm_model()
        tokenizer = load_tokenizer()
        max_length = 1440  

        st.write("Analyzing sentiment...")
        sentiment, confidence = predict_sentiment(review, lstm_model, tokenizer, max_length)

        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.write("Please enter a valid movie review.")

