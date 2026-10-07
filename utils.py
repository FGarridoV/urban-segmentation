import torch
import os
import pandas as pd

def get_device():
    """Detects and returns the best available hardware device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_optimal_batch_size(device):
    """Returns a conservative default batch size based on hardware."""
    if device == "cuda":
        return 8
    elif device == "mps":
        return 4
    return 1

def get_unprocessed_data(input_csv_path, output_csv_path):
    """
    Reads the input CSV, checks the output CSV (if it exists) to see which img_ids 
    are already processed, and returns a DataFrame of the remaining rows.
    """
    input_df = pd.read_csv(input_csv_path)
    
    if not os.path.exists(output_csv_path):
        return input_df, []
        
    try:
        output_df = pd.read_csv(output_csv_path)
        processed_ids = set(output_df['img_id'].values)
    except pd.errors.EmptyDataError:
        processed_ids = set()
        
    # Filtrar el dataframe para quedarnos solo con los img_id que no están en output_csv
    remaining_df = input_df[~input_df['img_id'].isin(processed_ids)].copy()
    
    return remaining_df, list(processed_ids)
