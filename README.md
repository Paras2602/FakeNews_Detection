# Fake News Detection System

This project is a Machine Learning–based application that classifies news articles as Fake or Real using Natural Language Processing (NLP) techniques.The system also provides explainability using LIME to highlight important words influencing predictions.

---

## Features
- Text preprocessing and cleaning  
- TF-IDF vectorization for feature extraction  
- Logistic Regression model for classification  
- Streamlit-based web application
- Simple and user-friendly interface
- Explainable predictions using LIME

---

## Technologies Used
- Python  
- Machine Learning  
- NLP (TF-IDF)  
- Streamlit
- Scikit-learn  

---

## Dataset
- Used a labeled fake news dataset (Fake/Real classification)  
- Data preprocessed before training  

---

## Model
- Algorithm Used: Logistic Regression
- Feature Extraction: TF-IDF  
- Trained using Scikit-learn  

---

## Project Structure
- `FakeNews_Detection.ipynb` – Model training and analysis  
- `app3.py` – Streamlit application  
- `fake_news_model.pkl` – Trained ML model  
- `requirements.txt` – Project dependencies  

---

## How to Run the Project

1. Clone the repository  
   ```bash
   git clone https://github.com/Paras2602/FakeNews_Detection.git
   cd FakeNews_Detection
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application  
   ```bash
   streamlit run app3.py
   ```

4. Open in browser  
   ```
   http://localhost:8501/
   ```

---

## Performance
- Achieved accuracy of ~85% on test dataset

## Output
<img width="937" height="627" alt="image" src="https://github.com/user-attachments/assets/77c3d211-08af-43a5-b580-7fa6141a0732" />
<img width="663" height="800" alt="image" src="https://github.com/user-attachments/assets/fc409488-72e0-4723-a9b0-71bb2c75086e" />
<img width="645" height="808" alt="image" src="https://github.com/user-attachments/assets/eda60927-853d-4adc-b6f9-26c25e656b3b" />
<img width="649" height="813" alt="image" src="https://github.com/user-attachments/assets/b097d05d-798f-4c43-9804-7146a6e6a537" />



---

## Future Improvements
- Improve model accuracy  
- Deploy on cloud  
- Add real-time news input  

---

## Author
Paras JB  
