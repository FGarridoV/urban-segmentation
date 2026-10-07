import os
import argparse
import pandas as pd
from tqdm import tqdm

from utils import get_device, get_optimal_batch_size, get_unprocessed_data
from models.mask2former import Mask2Former

def get_model(model_name: str, device: str):
    if model_name.lower() == "mask2former":
        return Mask2Former(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Batch Semantic Segmentation CLI")
    parser.add_argument("--model", type=str, required=True, help="Model acronym (e.g., mask2former)")
    parser.add_argument("--csv", type=str, required=True, help="Path to input metadata CSV")
    parser.add_argument("--img-dir", type=str, default="../images", help="Base directory for images")
    parser.add_argument("--out", type=str, default="results/segmentation_output.csv", help="Path to output CSV")
    parser.add_argument("--batch-size", type=int, default=None, help="Force a specific batch size")
    
    args = parser.parse_args()
    
    # 1. Hardware setup
    device = get_device()
    print(f"[*] Detected Hardware Device: {device.upper()}")
    
    batch_size = args.batch_size if args.batch_size is not None else get_optimal_batch_size(device)
    print(f"[*] Using Batch Size: {batch_size}")
    
    # 2. Checkpoint and data loading
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_remaining, processed_ids = get_unprocessed_data(args.csv, args.out)
    
    total_images = len(df_remaining) + len(processed_ids)
    print(f"[*] Total images in CSV: {total_images}")
    print(f"[*] Already processed: {len(processed_ids)}")
    print(f"[*] Remaining to process: {len(df_remaining)}")
    
    if len(df_remaining) == 0:
        print("[*] All images processed. Exiting.")
        return
        
    # 3. Model Initialization
    segmenter = get_model(args.model, device)
    
    # Retrieve all possible classes to ensure consistent CSV columns
    all_classes = list(segmenter.model.config.id2label.values())
    
    # Write header if new file
    if not os.path.exists(args.out):
        header = ["img_id"] + all_classes
        pd.DataFrame(columns=header).to_csv(args.out, index=False)
        
    # 4. Processing Loop
    # We iterate over the remaining dataframe in chunks (batches)
    records = df_remaining.to_dict('records')
    
    # Initialize tqdm progress bar
    pbar = tqdm(total=len(records), desc="Processing Images")
    
    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        
        valid_paths = []
        valid_ids = []
        
        for record in batch_records:
            img_path = os.path.join(args.img_dir, record['img_path'])
            if os.path.exists(img_path):
                valid_paths.append(img_path)
                valid_ids.append(record['img_id'])
            else:
                # Log missing images and write empty results
                print(f"[!] Warning: Image not found: {img_path}")
                # We can write an empty row or skip. Let's write an empty row (all 0.0) so it doesn't get re-evaluated
                empty_row = {"img_id": record['img_id']}
                for cls in all_classes:
                    empty_row[cls] = 0.0
                pd.DataFrame([empty_row]).to_csv(args.out, mode='a', header=False, index=False)
        
        if valid_paths:
            try:
                batch_results = segmenter.process_batch(valid_paths)
                
                # Format results for CSV appending
                rows_to_append = []
                for img_id, class_areas in zip(valid_ids, batch_results):
                    row = {"img_id": img_id}
                    for cls in all_classes:
                        row[cls] = class_areas.get(cls, 0.0)
                    rows_to_append.append(row)
                    
                # Append to CSV
                pd.DataFrame(rows_to_append).to_csv(args.out, mode='a', header=False, index=False)
                
            except Exception as e:
                print(f"\n[!] Error processing batch starting at {i}: {e}")
                # Save state by breaking or continuing. 
                # Since we write batch by batch, it's safe to break here.
                break
                
        pbar.update(len(batch_records))
        
    pbar.close()
    print("[*] Processing Complete!")

if __name__ == "__main__":
    main()
