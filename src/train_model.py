import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import dump
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack, csr_matrix
from src.preprocess import fit_vectorizer, transform_text, extract_image_features, get_numeric_features

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    return np.mean(diff) * 100

def train_model(train_path="data/train.csv"):
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(train_path)
    print(f"[+] Training data loaded → {df.shape}")

    y = df["price"].values

    # Text features
    vectorizer = fit_vectorizer(df)
    X_text = transform_text(df, vectorizer)

    # Image features
    X_img = extract_image_features(df)

    # Numeric features
    X_num, num_cols = get_numeric_features(df)

    # Combine all
    X_combined = hstack([
        X_text,
        csr_matrix(X_img),
        csr_matrix(X_num)
    ])

    print(f"[+] Combined feature shape: {X_combined.shape}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
    )
    model.fit(X_combined, y)

    preds = model.predict(X_combined)
    rmse = mean_squared_error(y, preds, squared=False)
    print(f"[✓] RMSE: {rmse:.4f} | SMAPE: {smape(y, preds):.2f}")

    dump(model, "models/model.pkl")
    dump(num_cols, "models/num_cols.pkl")
    print("[+] Model saved successfully.")

if __name__ == "__main__":
    train_model()
