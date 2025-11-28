# --- 1. NUCLEAR FIX: Force Installation of Dependencies ---
# This block runs BEFORE anything else and installs crucial libraries
import os
import sys
import subprocess
import nltk # NLTK must be imported before it can be used for downloads

# Set up installation list (Add missing pandas/numpy/requests/bs4)
REQUIRED_PKGS = [
    "joblib", 
    "lime==0.2.0.1", 
    "nltk", 
    "pandas", 
    "numpy", 
    "requests", 
    "beautifulsoup4",
    "scikit-learn" # Needed for the pipeline components
]

for pkg in REQUIRED_PKGS:
    # Use check_call to ensure installation succeeds before proceeding
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.DEVNULL)

# Download NLTK resources (runs every time the app starts, needed for cleaning)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- 2. Standard Imports (Now safe to import) ---
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
from bs4 import BeautifulSoup
import requests 

# --- 3. GLOBAL CLEANING SETUP (Must match training environment) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(raw_text):
    if not raw_text:
        return ""
    # REVISED CLEANING: This MUST match your training notebook exactly
    text = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# --- 4. MODEL LOADING ---
MODEL_PATH = 'fake_news_detector_pipeline.pkl' # Assuming you saved the pipeline under this name

model = None
try:
    # Load the full pipeline (Vectorizer + Classifier)
    model = joblib.load(MODEL_PATH) 
except Exception as e:
    # Display error if model fails to load
    st.error(f"❌ Failed to load model: {e}")
    st.info("Check if 'fake_news_detector_pipeline.pkl' is in your GitHub repo and correctly named.")
    st.stop()


# --- 5. STREAMLIT APP LOGIC ---
st.title("AI Fake News Detector")
st.write("Enter text or a URL to detect if it's fake news.")

# Initialize session state
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = "" # To store the raw text before prediction

option = st.selectbox("Input Type", ("Text", "URL"))

# --- URL Scraping Logic ---
if option == "URL":
    url = st.text_input("Enter news URL:")
    if st.button("Scrape"):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            raw_text = ' '.join(p.text for p in soup.find_all('p'))
            
            st.session_state.scraped_data = raw_text # Save raw text
            st.session_state.input_text = raw_text[:500] + "..." # Display preview
            st.info("Scraping successful. Click Detect to analyze.")
        except Exception as e:
            st.error(f"Scraping failed: {e}")
            st.session_state.scraped_data = ""

# --- Text Input Logic ---
elif option == "Text":
    st.session_state.scraped_data = st.text_area("Enter news text:", value=st.session_state.scraped_data)
    # Ensure text area updates session state directly for detection
    st.session_state.input_text = st.session_state.scraped_data

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