import gdown
import os

files_to_download = [
    {
        "url": "X",
        "output_folder": "data/raw",
        "filename": "amz_products_small.jsonl.gz",
    },
    {
        "url": "X",
        "output_folder": "models",
        "filename": "classifier_model.pkl",
    },
    {
        "url": "X",
        "output_folder": "data/processed",
        "filename": "X_train.pkl",
    },
]

for file in files_to_download:
    url = file["url"]
    output_folder = file["output_folder"]
    filename = file["filename"]
    
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, filename)
    
    print(f"Downloading {filename} to {output_folder}...")
    gdown.download(url, output_path, quiet=False)
    print(f"{filename} downloaded successfully to {output_folder}!")
