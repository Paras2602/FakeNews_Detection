import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
import requests
from bs4 import BeautifulSoup
import pymongo  # Optional for MongoDB — comment out if not using

# Load the trained model (ensure 'fake_news_model.pkl' is in the folder or load from URL if large)
try:
    model = joblib.load('fake_news_model.pkl')
except FileNotFoundError:
    st.error("Model file 'fake_news_model.pkl' not found! Ensure it's in the folder.")
    st.stop()

# Optional MongoDB connection (replace with your string; comment out if not using)
# client = pymongo.MongoClient("mongodb://localhost:27017/")  # Local example
# db = client['fake_news_db']
# collection = db['predictions']

# App Title
st.title("AI Fake News Detector")
st.write("Enter text or a URL to detect if it's fake news.")

# Input options
option = st.selectbox("Input Type", ("Text", "URL"))
input_text = ""

if option == "Text":
    input_text = st.text_area("Enter news text:")
elif option == "URL":
    url = st.text_input("Enter news URL:")
    if st.button("Scrape"):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            input_text = ' '.join(p.text for p in soup.find_all('p'))
            st.write("Scraped Text Preview:", input_text[:200])
        except Exception as e:
            st.error(f"Scraping failed: {e}")

# Detect if input is provided
if input_text and st.button("Detect"):
    # Predict
    prediction = model.predict([input_text])[0]
    prob = model.predict_proba([input_text])[0][1]  # Prob of Fake

    label = "Fake" if prediction == 1 else "Real"
    st.subheader("Result:")
    st.success(f"{label} (Confidence: {prob*100:.2f}%)")

    # Explain with LIME
    explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
    exp = explainer.explain_instance(input_text, model.predict_proba, num_features=10)
    st.subheader("Explanation (Why it's Fake/Real):")
    st.write(exp.as_list())

    # Optional: Save to MongoDB (comment out if not using)
    # collection.insert_one({"text": input_text, "label": label, "confidence": prob})
    # st.info("Prediction saved to database.")