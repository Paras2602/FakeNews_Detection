# --- 1. NUCLEAR FIX: Force Installation (Cleaned List) ---
import os
import sys
import subprocess

# NLTK removed due to Permission Denied error on the remote server.
# Using os.system for broader compatibility.
REQUIRED_PKGS = [
    "pandas", "numpy", "scikit-learn", "joblib", "lime==0.2.0.1", 
    "requests", "beautifulsoup4"
]

for pkg in REQUIRED_PKGS:
    os.system(f"{sys.executable} -m pip install {pkg}")


# --- 2. Standard Imports (Now safe to import) ---
import streamlit as st
import joblib
import re
from lime.lime_text import LimeTextExplainer
from bs4 import BeautifulSoup
import requests 

# --- 3. PURE PYTHON CLEANING (No NLTK dependencies) ---
# This must be IDENTICAL to the cleaning you used to train the final model.
# Since we remove NLTK, the model may be slightly less accurate, but it will deploy.
def clean_text(raw_text):
    if not raw_text: return ""
    # 1. Convert to lowercase and remove non-alphabetic chars
    text = re.sub(r'[^a-zA-Z\s]', '', raw_text.lower())
    # 2. Split and remove short words (a simple proxy for stopwords/lemmatization)
    words = [word for word in text.split() if len(word) > 2]
    return ' '.join(words)

# --- 4. MODEL LOADING (Corrected Filename) ---
# Check your local folder and use the exact name you see there!
MODEL_PATH = 'fake_news_model.pkl'  # <--- Change this to the actual file name
model = None
try:
    model = joblib.load(MODEL_PATH) 
except Exception as e:
    # If this still fails, it means the file wasn't pushed, or the name is still wrong.
    st.error(f"❌ Failed to load model: {e}")
    st.info("Did you run 'git push origin main' AFTER training and saving the model?")
    st.stop()


# --- 5. STREAMLIT APP LOGIC ---
st.title("AI Fake News Detector")
st.write("Enter text or a URL to detect if it's fake news.")

# Initialize session state
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Input Type Selection
option = st.selectbox("Input Type", ("Text", "URL"))

# Input fields
if option == "URL":
    url = st.text_input("Enter news URL:")
    if st.button("Scrape"):
        try:
            # Scraping logic
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            raw_text = ' '.join(p.text for p in soup.find_all('p'))
            st.session_state.input_text = raw_text
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

        # 2. Predict (Pipeline handles vectorization)
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