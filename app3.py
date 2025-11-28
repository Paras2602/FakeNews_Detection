import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
import requests
from bs4 import BeautifulSoup
import pymongo  # Optional

# Load model
model = joblib.load('fake_news_model.pkl')

# Optional MongoDB (comment out if not using)
# client = pymongo.MongoClient("mongodb://localhost:27017/")
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
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        input_text = ' '.join(p.text for p in soup.find_all('p'))
        st.write("Scraped Text Preview:", input_text[:200])

# Detect button
if input_text and st.button("Detect"):
    prediction = model.predict([input_text])[0]
    prob = model.predict_proba([input_text])[0][1]  # Prob of Fake

    label = "Fake" if prediction == 1 else "Real"
    st.subheader("Result:")
    st.success(f"{label} (Confidence: {prob*100:.2f}%)")

    # Explain with LIME
    explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
    exp = explainer.explain_instance(input_text, model.predict_proba, num_features=10)
    st.subheader("Explanation:")
    st.write(exp.as_list())

    # Optional: Save to MongoDB
    # collection.insert_one({"text": input_text, "label": label, "confidence": prob})
    # st.info("Prediction saved to database.")