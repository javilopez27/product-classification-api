import pandas as pd
import numpy as np
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
from utils.data_loader import load_data

def preprocess_data(input_path, output_dir, limit=None, seed=42):
    """
    Preprocess the data for use in the classification model.
    
    Args:
        input_path (str): Path to the compressed JSONL file.
        output_dir (str): Directory to save processed data.
        limit (int, optional): Maximum number of rows to load for testing.
        seed (int, optional): Seed for random selection (for reproducibility).
    """
    # Load data with limit and random selection (if specified)
    df = load_data(input_path, limit=limit, seed=seed)
    
    # Process text
    df['text'] = (
        df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' +
        df['title'].fillna('') + ' ' +
        df['feature'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    )
    
    # Handle missing data. Even if there are no missing values, I treat it as a minimum.
    df['brand'] = df['brand'].fillna('Unknown')
    df['price'] = df['price'].fillna('0')
    df = df[df['main_cat'].notna()]  # Filter products without a main category

    # Preprocess `price` column
    df['price'] = (
        df['price']
        .str.strip()  # Remove whitespace
        .str.replace('[\$,]', '', regex=True)  # Remove symbols like $ and commas
        .replace(r'[^\d\.]', '', regex=True)  # Remove any non-numeric or non-decimal characters
        .replace('', np.nan)  # Replace empty strings with NaN
    )

    # Convert to float and handle errors
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    # Handle NaN and outliers in `price`
    df['price'] = df['price'].fillna(0)
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df['price_cleaned'] = df['price'].apply(
        lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x)
    )
    df['price_log'] = df['price_cleaned'].apply(lambda x: np.log(x + 1))

    # Scale prices between 0 and 1
    scaler = MinMaxScaler()
    df['price_normalized'] = scaler.fit_transform(df[['price_cleaned']])

    # Save the scaler
    joblib.dump(scaler, "metadata/price_scaler.pkl")

    # Save outlier limits
    price_metadata = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }
    with open("metadata/price_metadata.json", "w") as f:
        json.dump(price_metadata, f)

    # Create a column to indicate if images are available
    df['has_image'] = df['image'].apply(lambda x: 0 if len(x) == 0 else 1)

    # Process relationships in `also_buy` and `also_view`
    df['also_buy_count'] = df['also_buy'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['also_view_count'] = df['also_view'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Vectorize text with TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000)
    X_text = vectorizer.fit_transform(df['text'])
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # Frequency encoding for 'brand'
    freq_encoded = df['brand'].value_counts(normalize=True)
    df['brand_encoded'] = df['brand'].map(freq_encoded)

    # Convert tabular features to a dense matrix
    other_features = df[['brand_encoded', 'price_normalized', 'has_image', 'also_buy_count', 'also_view_count']].values

    # Combine sparse matrix with other features
    X = hstack([X_text, other_features])      
    y = df['main_cat']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save processed data
    joblib.dump(X_train, f"{output_dir}/X_train.pkl")  # Save as sparse matrix for efficiency
    joblib.dump(X_test, f"{output_dir}/X_test.pkl")
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_test.npy", y_test)

if __name__ == "__main__":
    # preprocess_data("data/raw/amz_products_small.jsonl.gz", "data/processed")
    preprocess_data("data/raw/amz_products_small.jsonl.gz", "data/processed", limit=100000)  # For testing
