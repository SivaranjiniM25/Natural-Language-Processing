import streamlit as st
import pickle
from preprocessing import clean_text

model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Movie Review Sentiment Analysis")
user_input = st.text_area('Enter your review:')

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        
        prediction = model.predict(vector)[0]
        
        probabilities = model.predict_proba(vector)[0]
        confidence = probabilities[prediction] * 100

        if prediction == 1:
            st.success(f"Positive (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"Negative (Confidence: {confidence:.2f}%)")
            
        # Show a progress bar for visual impact
        st.progress(int(confidence))