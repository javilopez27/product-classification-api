import gzip
import json
import pandas as pd
import random

def parse_data(path):
    """Reads a compressed JSONL file line by line."""
    with gzip.open(path, 'r') as f:
        for line in f:
            yield json.loads(line)

def load_data(path, limit=None, seed=42):
    """
    Loads a JSONL dataset into a pandas DataFrame, randomly selecting rows if a limit is applied.
    
    Args:
        path (str): Path to the .jsonl.gz file.
        limit (int, optional): Maximum number of rows to load for testing.
        seed (int, optional): Seed for random selection reproducibility.
    
    Returns:
        pd.DataFrame: Dataset loaded as a DataFrame.
    """
    if limit is None:
        # Load all rows if no limit is applied
        return pd.DataFrame(parse_data(path))
    
    random.seed(seed)  # Set a seed for reproducibility
    selected_data = []
    total_count = 0

    # Iterate line by line and randomly select rows until the limit is reached
    for record in parse_data(path):
        total_count += 1
        if len(selected_data) < limit:
            selected_data.append(record)
        else:
            # Replace an element with probability `limit / total_count`
            idx = random.randint(0, total_count - 1)
            if idx < limit:
                selected_data[idx] = record

    return pd.DataFrame(selected_data)
