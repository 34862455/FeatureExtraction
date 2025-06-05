import gzip
import pickle
import torch

# Number of samples to extract
n_samples = 5

# File paths
base_path = "/home/minneke/Documents/Projects/SignExperiments.old/data/"
paths = {
    "dev": base_path + "DSG_dev.pt",
    "test": base_path + "DSG_test.pt",
    "train": base_path + "DSG_train.pt"
}

for split, path in paths.items():
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    subset = data[:n_samples]

    # Save subset as gzipped pickle again
    out_path = base_path + f"DSG_{split}_subset.pt"
    with gzip.open(out_path, "wb") as f:
        pickle.dump(subset, f)

    print(f"Saved {n_samples} samples to {out_path}")
