import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image
import requests
from io import BytesIO
from scipy.sparse import hstack, csr_matrix
from src.preprocess import clean_text, get_image_model, load_image_from_link, load_vectorizer

st.set_page_config(page_title="Smart Product Pricing", layout="wide")
st.title("🧠 Smart Product Pricing — Text + Image + Numeric Model")

st.sidebar.header("📦 Bulk Prediction")
uploaded = st.sidebar.file_uploader("Upload test.csv", type=["csv"])

if uploaded:
    with open("data/test_uploaded.csv", "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("✅ Test file uploaded.")
    if st.sidebar.button("Run Predictions"):
        from src.predict import predict_prices
        df_pred = predict_prices("data/test_uploaded.csv")
        # if uploaded file contains quantity column, include totals
        try:
            df_uploaded = pd.read_csv("data/test_uploaded.csv")
            if "quantity" in df_uploaded.columns:
                qty = df_uploaded["quantity"].fillna(1).astype(int)
                df_pred["quantity"] = qty.values
                df_pred["total_price"] = df_pred["price"] * df_pred["quantity"]
        except Exception:
            pass
        st.dataframe(df_pred.head(10))
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("Download submission.csv", data=csv, file_name="submission.csv")

st.header("🎯 Predict Single Product")

desc = st.text_area("Enter product description (catalog_content):")
img_link = st.text_input("Enter image URL or path (optional):")
num_inputs = st.text_input("Enter numeric feature values (comma-separated): e.g. 4.5, 100, 20")
quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

if st.button("Predict Price"):
    # Load artifacts with safe fallbacks
    try:
        vectorizer = load_vectorizer("models/vectorizer.pkl")
    except Exception:
        st.error("Could not load vectorizer (models/vectorizer.pkl).")
        st.stop()

    try:
        model = load("models/model.pkl")
    except Exception:
        st.error("Could not load model (models/model.pkl).")
        st.stop()

    # load num_cols safely
    try:
        num_cols = load("models/num_cols.pkl")
    except Exception:
        num_cols = []
    if num_cols is None:
        num_cols = []
    # ensure num_cols is a list-like
    if not isinstance(num_cols, (list, tuple, np.ndarray)):
        try:
            num_cols = list(num_cols)
        except Exception:
            num_cols = []

    # --- text ---
    try:
        cleaned = clean_text(desc) if desc is not None else ""
        X_text = vectorizer.transform([cleaned])
    except Exception:
        st.warning("Text transformation failed — using empty text.")
        X_text = vectorizer.transform([""])

    # --- image ---
    X_img = None
    try:
        model_cnn = get_image_model()
    except Exception:
        model_cnn = None

    if img_link.strip():
        if model_cnn is not None:
            try:
                img_tensor = load_image_from_link(img_link)
                emb = model_cnn.predict(img_tensor, verbose=0).flatten()
                X_img = np.expand_dims(emb, axis=0)
            except Exception:
                X_img = None
                st.warning("Could not process image. Using zeros for image features.")
        else:
            X_img = None
            st.warning("Image model not available. Using zeros for image features.")
    # default image vector (size matches typical embedding or 0-length if unknown)
    if X_img is None:
        # try to infer feature size from saved models or default to 2048
        img_feat_size = 0
        try:
            # if model provides output shape
            if model_cnn is not None and hasattr(model_cnn, "output_shape"):
                out_shape = model_cnn.output_shape
                if isinstance(out_shape, tuple):
                    img_feat_size = int(np.prod([s for s in out_shape if s is not None][1:])) if len(out_shape) > 1 else int(out_shape[-1])
        except Exception:
            img_feat_size = 0
        if img_feat_size <= 0:
            img_feat_size = 2048
        X_img = np.zeros((1, img_feat_size))

    # --- numeric ---
    # if no numeric columns known, create empty array with shape (1,0)
    n_num = len(num_cols)
    if n_num == 0:
        X_num = np.zeros((1, 0))
    else:
        if not num_inputs or not num_inputs.strip():
            X_num = np.zeros((1, n_num))
        else:
            items = [s.strip() for s in num_inputs.split(",")]
            items = [s for s in items if s != ""]
            nums = []
            for s in items:
                try:
                    nums.append(float(s))
                except Exception:
                    nums.append(0.0)
            nums = np.array(nums, dtype=float)
            if nums.size < n_num:
                nums = np.pad(nums, (0, n_num - nums.size), constant_values=0.0)
            elif nums.size > n_num:
                nums = nums[:n_num]
            X_num = nums.reshape(1, -1)

    # --- combine ---
    try:
        X_combined = hstack([X_text, csr_matrix(X_img), csr_matrix(X_num)])
    except Exception as e:
        st.error(f"Could not combine features: {e}")
        st.stop()

    try:
        unit_price = model.predict(X_combined)[0]
        total_price = unit_price * int(quantity)
        st.success(f"💰 Predicted Unit Price: ₹{unit_price:.2f}")
        st.info(f"🧾 Quantity: {int(quantity)}  →  Total Price: ₹{total_price:.2f}")
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    if img_link.strip():
        try:
            if img_link.startswith("http"):
                response = requests.get(img_link, timeout=5)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(img_link)
            st.image(img, caption="Product Image", width=250)
        except Exception:
            st.warning("Could not display image.")