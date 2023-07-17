import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
# Load the saved model
with open("naive_bayes_model.pkl", "rb") as file:a
    model = pickle.load(file)
# Load the saved TfidfVectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
# Streamlit app code
st.title("Sentiment Analysis App")
st.markdown("By Sajid")
image = Image.open("sentiment.png")
st.image(image, use_column_width=True)

st.subheader("Enter your text here:")
user_input = st.text_area("")
# Create a predict button
if st.button("Predict"):
    # Preprocess the input text
    text_vectorized = vectorizer.transform([user_input])
    # Make predictions
    prediction = model.predict(text_vectorized)[0]
    st.header("Prediction:")
    # Display the predicted sentiment
    if prediction == 'negative':
        st.subheader("The sentiment of given text is: Negative")
    elif prediction == 'neutral':
        st.subheader("The sentiment of given text is: Neutral")
    elif prediction == 'positive':
        st.subheader("The sentiment of given text is: Positive")