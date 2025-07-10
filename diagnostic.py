import torch

# Replace with your actual paths
ckpt_a = torch.load("checkpoints/finetuning/epoch07.pt", map_location='cpu')
ckpt_b = torch.load("checkpoints/finetuning/s3d_classifier_ft_epoch09.pt", map_location='cpu')

state_a = ckpt_a["model_state_dict"]
state_b = ckpt_b["model_state_dict"]

# print(f"Model state:")
# for k, d in state_a[:10]:
#     print(f"  {k}: {d:.6f}")

unchanged = []
changed = []
for k in state_a:
    if k in state_b:
        diff = (state_a[k] - state_b[k]).abs().sum().item()
        if diff < 1e-6:
            unchanged.append(k)
        else:
            changed.append((k, diff))

print(f"Total layers: {len(state_a)}")
print(f"Changed layers: {len(changed)}")
print("Examples of changed layers (with L1 diff):")
for k, d in changed[10:]:
    print(f"  {k}: {d:.6f}")

# import pickle
# import gzip
#
# # Path to the saved feature file
# path = "data/DSG_classifier09_dev.pt"  # or train/test depending on what you want
#
# # Load the file
# with gzip.open(path, "rb") as f:
#     data = pickle.load(f)
#
# # Inspect the shape of the first sample
# print("Number of samples:", len(data))
# print("Shape of 'sign' features in first sample:", data[0]["sign"].shape)
