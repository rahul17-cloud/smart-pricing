import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image
from io import BytesIO
import requests

# ---------------- TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fit_vectorizer(df, text_col="catalog_content", save_path="models/vectorizer.pkl"):
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    vectorizer.fit(df[text_col])
    dump(vectorizer, save_path)
    print(f"[+] TF-IDF vectorizer saved → {save_path}")
    return vectorizer

def load_vectorizer(path="models/vectorizer.pkl"):
    return load(path)

def transform_text(df, vectorizer, text_col="catalog_content"):
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return vectorizer.transform(df[text_col])

# ---------------- IMAGE ----------------
def get_image_model():
    base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base.input, outputs=base.output)
    return model

def load_image_from_link(link, size=(224, 224)):
    try:
        if isinstance(link, str) and link.startswith("http"):
            response = requests.get(link, timeout=5)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(link)
        img = img.convert("RGB").resize(size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    except Exception:
        return np.zeros((1, size[0], size[1], 3))

def extract_image_features(df, image_col="image_link"):
    model = get_image_model()
    feats = []
    for link in tqdm(df[image_col], desc="Extracting image embeddings"):
        img_tensor = load_image_from_link(link)
        emb = model.predict(img_tensor, verbose=0)
        feats.append(emb.flatten())
    return np.vstack(feats)

# ---------------- NUMERIC ----------------
def get_numeric_features(df, exclude_cols=["sample_id", "catalog_content", "image_link", "price"]):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = [c for c in num_cols if c not in exclude_cols]
    if not num_cols:
        print("[!] No numeric columns found.")
        return np.empty((len(df), 0)), None
    print(f"[+] Numeric features used: {num_cols}")
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])
    dump(scaler, "models/scaler.pkl")
    return X_num, num_cols

def transform_numeric(df, num_cols):
    if not num_cols:
        return np.empty((len(df), 0))
    scaler = load("models/scaler.pkl")
    return scaler.transform(df[num_cols])
