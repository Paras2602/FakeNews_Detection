# --- 1. NUCLEAR FIX: Sequential and Explicit Installation ---
# This block runs BEFORE anything else and installs crucial libraries sequentially
import os
import sys
import subprocess

# List core dependencies that MUST be force-installed
REQUIRED_PKGS = [
    "pandas",
    "numpy", 
    "scikit-learn", 
    "joblib", 
    "nltk",
    "lime==0.2.0.1", 
    "requests", 
    "beautifulsoup4"
]

# Run pip install command for required packages
for pkg in REQUIRED_PKGS:
    try:
        # Use simple system call for better compatibility on some Streamlit runners
        os.system(f"{sys.executable} -m pip install {pkg}")
    except Exception as e:
        print(f"Failed to force install {pkg}: {e}")

# --- 2. Standard Imports (Now safe to import) ---
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
from bs4 import BeautifulSoup
import requests
import nltk # NLTK is now installed


# --- 3. NLTK Download (Required resources) ---
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass # Continue even if download fails once


# --- 4. GLOBAL CLEANING SETUP (Must match training environment) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(raw_text):
    if not raw_text: return ""
    text = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# --- 5. MODEL LOADING ---
MODEL_PATH = 'fake_news_detector_pipeline.pkl' 
model = None
try:
    model = joblib.load(MODEL_PATH) 
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


# --- 6. STREAMLIT APP LOGIC ---
st.title("AI Fake News Detector")
st.write("Enter text or a URL to detect if it's fake news.")

# Initialize session state
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = "" 

# Input Type Selection
option = st.selectbox("Input Type", ("Text", "URL"))

# Input fields
if option == "URL":
    url = st.text_input("Enter news URL:")
    if st.button("Scrape"):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            raw_text = ' '.join(p.text for p in soup.find_all('p'))
            st.session_state.input_text = raw_text # Use input_text for raw storage
            st.info("Scraping successful. Click Detect to analyze.")
        except Exception as e:
            st.error(f"Scraping failed: {e}")
            st.session_state.input_text = ""
            
elif option == "Text":
    st.session_state.input_text = st.text_area("Enter news text:", value=st.session_state.input_text)


# --- Prediction Logic ---
if st.button("Detect"):
    text_to_analyze = st.session_state.input_text

    if not text_to_analyze or len(text_to_analyze) < 50:
        st.warning("Please enter at least 50 characters of text to analyze.")
        st.stop()

    try:
        # 1. Clean the text using the trained function
        cleaned_input = clean_text(text_to_analyze)

        # 2. Predict (Pipeline automatically vectorizes the clean text)
        prediction = model.predict([cleaned_input])[0]
        prob_array = model.predict_proba([cleaned_input])[0]
        
        # 3. Label Mapping (Assuming 0=Real, 1=Fake)
        label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        confidence = prob_array[prediction] * 100

        st.subheader("🔍 Prediction Result:")
        if label == "FAKE NEWS":
            st.error(f"❌ This article is likely **{label}** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ This article is likely **{label}** (Confidence: {confidence:.2f}%)")

        # --- LIME Explanation ---
        explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
        exp = explainer.explain_instance(cleaned_input, model.predict_proba, num_features=10)
        
        st.subheader("💡 Key Influencing Words:")
        for word, weight in exp.as_list():
            color = "green" if weight < 0 else "red"
            st.markdown(f'<span style="color:{color}; font-weight: bold;">{word}</span> (Weight: {weight:.3f})', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")