import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from preprocessing import clean_unrealistic_values, preprocess_data
from clustering_algorithms.clustering_algorithms import run_clustering
from typing import Optional, Tuple, List, Any, Dict
import sys

def parse_param_value(val: str) -> Any:
    """Convert a string to int, float, or leave as string."""
    try:
        if '.' in val:
            return float(val)
        return int(val)
    except ValueError:
        return val

def evaluate(X: np.ndarray, labels: np.ndarray) -> None:
    """
    Calculate and print the silhouette score for the clustering labels.

    Args:
        X (np.ndarray): Feature matrix.
        labels (np.ndarray): Cluster labels.
    """
    unique_labels = set(labels.tolist())
    if len(unique_labels) > 1:
        score = silhouette_score(X, labels)
        print(f"Silhouette Score: {score:.4f}")
    else:
        print(f"Silhouette not defined for clusters: {unique_labels}")


def main() -> None:
    """
    Main function to run the clustering pipeline:
      1. Load data
      2. Clean unrealistic values
      3. Preprocess (scale & encode)
      4. Run clustering
      5. Evaluate silhouette score
      6. Save cluster assignments
    """
    parser = argparse.ArgumentParser(
        description="Run clustering on CSV with evaluation",
        epilog="For more details and examples, see GUIDE.md in the project directory."
    )
    parser.add_argument('csv_file', type=str, help='Input CSV file path')
    parser.add_argument('algorithm', type=str, nargs='?', default=None, help='Clustering algorithm key (optional, default: agglomerative_average)')
    parser.add_argument('--clean', action='store_true', help='Bypass cleaning of unrealistic values')
    # Parse known args and allow additional key=value algorithm params
    args, unknown = parser.parse_known_args()
    # Extract additional algorithm parameters
    params: Dict[str, Any] = {}
    for item in unknown:
        if '=' in item:
            key, val = item.split('=', 1)
            params[key] = parse_param_value(val)
    if params:
        print(f"Parsed algorithm parameters: {params}")

    # Determine algorithm (default: agglomerative_average)
    if args.algorithm:
        algorithm = args.algorithm
    else:
        algorithm = 'agglomerative_average'
        print("[INFO] No algorithm specified. Defaulting to hierarchical clustering with average linkage ('agglomerative_average'). See GUIDE.md for details.")

    # STEP 1: Load and inspect raw data
    print("\nSTEP 1: Loading raw data")
    raw_df = pd.read_csv(args.csv_file)
    print(f"Loaded raw data: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # STEP 2: Clean unrealistic values
    if args.clean:
        print("\nSTEP 2: Cleaning unrealistic values (as requested)")
        clean_df = clean_unrealistic_values(raw_df)
        removed = raw_df.shape[0] - clean_df.shape[0]
        print(f"Removed {removed} unrealistic rows; now {clean_df.shape[0]} rows")
    else:
        print("\nSTEP 2: Skipping cleaning of unrealistic values (default). See GUIDE.md for details on cleaning.")
        clean_df = raw_df

    # STEP 3: Preprocess data (scaling + encoding)
    print("\nSTEP 3: Preprocessing data (scaling + encoding)")
    df_proc, num_cols, enc_cols, scaler, patient_ids = preprocess_data(clean_df)
    print("Data preprocessing (scaling + encoding) completed")

    # STEP 4: Run clustering
    print("\nSTEP 4: Running clustering")
    X = df_proc.values  # type: np.ndarray
    labels, model, metrics = run_clustering(X, algorithm=algorithm, **params)
    print(f"\nClustering metrics: {metrics}")

    # STEP 5: Evaluate clustering
    print("\nSTEP 5: Evaluating silhouette score")
    evaluate(X, labels)

    # STEP 6: Save cluster assignments to CSV
    print("\nSTEP 6: Saving cluster assignments")
    out_df = pd.DataFrame({'cluster_id': labels})
    if patient_ids is not None:
        out_df.insert(0, 'patient_id', patient_ids.reset_index(drop=True))
    out_file = f"{algorithm}_clusters.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Saved cluster assignments to {out_file}")


if __name__ == '__main__':  # pragma: no cover
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please check your input file path and refer to GUIDE.md for correct usage and examples.")
    except SyntaxError as e:
        print(f"\n[ERROR] Syntax error: {e}")
        print("Please check your command syntax and refer to GUIDE.md for correct usage and examples.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("An error occurred. Please refer to GUIDE.md for correct usage and troubleshooting.")
