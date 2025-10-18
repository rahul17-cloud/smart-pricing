# Smart Pricing Project

## Overview
The Smart Pricing project is designed to provide dynamic pricing solutions using various algorithms. It includes a service for calculating prices, a controller for handling requests, and routes for API endpoints.

## Features
- Calculate prices based on different algorithms.
- API endpoints for retrieving and updating pricing information.
- Unit tests to ensure the accuracy of pricing calculations.

## File Structure
```
smart-pricing/
├─ data/
│  ├─ train.csv
│  └─ test.csv
├─ src/
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ predict.py
├─ models/
│  ├─ model.pkl
│  ├─ vectorizer.pkl
│  ├─ image_encoder.pkl
│  └─ scaler.pkl
├─ output/
│  └─ submission.csv
└─ app/
   └─ streamlit_app.py

