from tqdm import tqdm
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_model(data_dir, model_path):
    """
    Trains a classification model to predict `main_cat`.
    
    Args:
        data_dir (str): Directory containing the processed data.
        model_path (str): Path to save the trained model.
    """
    # Load processed data
    X_train = joblib.load(f"{data_dir}/X_train.pkl")  # Load sparse matrix
    y_train = np.load(f"{data_dir}/y_train.npy", allow_pickle=True)
    X_test = joblib.load(f"{data_dir}/X_test.pkl")
    y_test = np.load(f"{data_dir}/y_test.npy", allow_pickle=True)

    # Train a model (Random Forest with progress bar)
    model = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)

    # Train with tqdm for progress bar
    for i in tqdm(range(1, model.n_estimators + 1), desc="Training Progress"):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    train_model("data/processed", "models/classifier_model.pkl")
