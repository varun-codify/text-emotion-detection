import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

from src.preprocess import clean_text

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/emotions.csv")

    print("Cleaning text...")
    df["clean"] = df["text"].apply(clean_text)

    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean"]).toarray()
    y = df["emotion"]

    print("Training model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    print("Saving model...")
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Training Completed.")

if __name__ == "__main__":
    main()
