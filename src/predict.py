import os
import numpy as np
import pandas as pd
from joblib import load
from scipy.sparse import hstack, csr_matrix
from src.preprocess import load_vectorizer, transform_text, extract_image_features, transform_numeric

def predict_prices(test_path="data/test.csv"):
    os.makedirs("output", exist_ok=True)
    df = pd.read_csv(test_path)
    print(f"[+] Test data loaded → {df.shape}")

    model = load("models/model.pkl")
    vectorizer = load_vectorizer("models/vectorizer.pkl")
    num_cols = load("models/num_cols.pkl")

    X_text = transform_text(df, vectorizer)
    X_img = extract_image_features(df)
    X_num = transform_numeric(df, num_cols)

    X_combined = hstack([
        X_text,
        csr_matrix(X_img),
        csr_matrix(X_num)
    ])

    preds = model.predict(X_combined)
    df_out = pd.DataFrame({
        "sample_id": df["sample_id"],
        "price": preds
    })
    df_out.to_csv("output/submission.csv", index=False)
    print("[✓] Predictions saved → output/submission.csv")
    return df_out

if __name__ == "__main__":
    predict_prices()
