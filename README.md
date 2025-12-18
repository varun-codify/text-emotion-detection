# Fake News Detection using NLP and Machine Learning

## 📌 Overview
This project implements an NLP-based Fake News Detection system that classifies news articles as **REAL** or **FAKE** using Machine Learning techniques. The system applies text preprocessing, TF-IDF feature extraction, and a Logistic Regression classifier, along with a Streamlit web interface for real-time prediction.

## 🎯 Problem Statement
With the rapid spread of misinformation online, identifying fake news has become critical. This project aims to automatically detect fake news articles based on textual content.

## 🧠 Model Pipeline
1. Text preprocessing (cleaning & normalization)
2. TF-IDF vectorization
3. Logistic Regression classification
4. Prediction with confidence score

## ⚙️ Technologies Used
- Python  
- Scikit-learn  
- Natural Language Processing (NLP)  
- TF-IDF Vectorizer  
- Logistic Regression  
- Streamlit  

## 📂 Project Structure
fake-news-detection/
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── predict.py
│ └── utils.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


## 📊 Dataset
This project uses the **Fake and Real News Dataset** from Kaggle.  
The dataset is **not included** in this repository due to size constraints and best practices.

👉 You can download it from Kaggle and place it inside a `data/` folder.

## 🚀 How to Run

pip install -r requirements.txt
python src/train.py
streamlit run app.py

