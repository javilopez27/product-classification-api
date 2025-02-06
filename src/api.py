from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import hstack
import json
import pickle

# Load model, vectorizer, scaler and metadata
MODEL_PATH = "models/classifier_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
SCALER_PATH = "metadata/price_scaler.pkl"
METADATA_PATH = "metadata/price_metadata.json"


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)

# Load price metadata
with open(METADATA_PATH, "r") as f:
    price_metadata = json.load(f)
lower_bound = price_metadata["lower_bound"]
upper_bound = price_metadata["upper_bound"]

# Create FastAPI application
app = FastAPI()

# Input data model
class ProductData(BaseModel):
    description: list
    title: str
    feature: list
    brand: str
    price: float = 0.0
    image: list = []
    also_buy: list = []
    also_view: list = []

@app.post("/predict")
def predict(data: ProductData):
    try:
        # Preprocess text
        text = (
            ' '.join(data.description if isinstance(data.description, list) else []) + ' ' +
            data.title + ' ' +
            ' '.join(data.feature if isinstance(data.feature, list) else [])
        )

        # Transform text using the vectorizer
        text_vectorized = vectorizer.transform([text])

        # Encode brand
        brand_freq = getattr(vectorizer, "brand_freq", {})
        brand_encoded = brand_freq.get(data.brand, 0.0)

        # Normalize price
        price = data.price
        price_cleaned = max(min(price, upper_bound), lower_bound)  # Apply outlier limits
        price_normalized = scaler.transform([[price_cleaned]])[0][0]  # Normalize using the scaler

        # Check if it has an image
        has_image = 1 if len(data.image) > 0 else 0

        # Count also_buy and also_view relationships
        also_buy_count = len(data.also_buy)
        also_view_count = len(data.also_view)

        # Combine sparse and dense features
        features_dense = np.array([[brand_encoded, price_normalized, has_image, also_buy_count, also_view_count]])
        features = hstack([text_vectorized, features_dense])

        # Perform prediction
        prediction = model.predict(features)[0]

        return {"main_cat": prediction}

    except Exception as e:
        return {"error": str(e)}
