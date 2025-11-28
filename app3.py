import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
import requests
from bs4 import BeautifulSoup
import pymongo  # Optional for MongoDB — comment out if not using

# Initialize session state to save input text between reruns
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Load the trained model (ensure 'fake_news_model.pkl' is in the folder)
try:
    model = joblib.load('fake_news_model.pkl')
except FileNotFoundError:
    st.error("Model file 'fake_news_model.pkl' not found! Ensure it's in the folder.")
    st.stop()

# Optional MongoDB connection (comment out if not using)
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client['fake_news_db']
# collection = db['predictions']

# Simple text cleaning (matches your preprocessing)
def clean_text(text):
    text = text.lower()
    text = ' '.join(word for word in text.split() if word.isalpha())
    return text

# App Title
st.title("AI Fake News Detector")
st.write("Enter text or a URL to detect if it's fake news.")

# Input options
option = st.selectbox("Input Type", ("Text", "URL"))

if option == "Text":
    st.session_state.input_text = st.text_area("Enter news text:", value=st.session_state.input_text)
elif option == "URL":
    url = st.text_input("Enter news URL:")
    if st.button("Scrape"):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            scraped_text = ' '.join(p.text for p in soup.find_all('p'))
            st.session_state.input_text = clean_text(scraped_text)
            st.write("Scraped Text Preview:", st.session_state.input_text[:200])
        except Exception as e:
            st.error(f"Scraping failed: {e}")

# Detect button (uses session state to keep input text)
if st.button("Detect"):
    if st.session_state.input_text:
        try:
            # Predict
            prediction = model.predict([st.session_state.input_text])[0]
            prob = model.predict_proba([st.session_state.input_text])[0][1]  # Probability of Fake

            label = "Fake" if prediction == 1 else "Real"
            st.subheader("Result:")
            st.success(f"{label} (Confidence: {prob*100:.2f}%)")

            # Explain with LIME
            explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
            exp = explainer.explain_instance(st.session_state.input_text, model.predict_proba, num_features=10)
            st.subheader("Explanation (Why it's Fake/Real):")
            st.write(exp.as_list())

            # Optional: Save to MongoDB
            # collection.insert_one({"text": st.session_state.input_text, "label": label, "confidence": prob})
            # st.info("Prediction saved to database.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please enter text or scrape a URL first.")