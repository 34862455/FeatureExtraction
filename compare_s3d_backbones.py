from model_s3d import S3D
from s3d_ctc_finetune import S3DRecognizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load original pretrained S3D (Kinetics-400) ===
s3d_pre = S3D(num_class=400)
weights_pre = torch.load("checkpoints/S3D_kinetics400.pt", map_location=device)

if any(k.startswith("module.") for k in weights_pre.keys()):
    print("Detected 'module.' prefix in pretrained model")
    weights_pre = {k.replace("module.", ""): v for k, v in weights_pre.items()}

missing, unexpected = s3d_pre.load_state_dict(weights_pre, strict=False)
print(f"[PRETRAINED] Missing: {missing}, Unexpected: {unexpected}")
s3d_pre.replace_logits(None)  # remove classification head

# === Sanity Check: Pretrained model weights ===
with torch.no_grad():
    w = next(iter(s3d_pre.state_dict().values()))
    print(f"=== Pretrained S3D ===")
    print(f"mean: {w.mean().item():.6f}, std: {w.std().item():.6f}, min: {w.min().item():.6f}, max: {w.max().item():.6f}")

# === Load fine-tuned model ===
s3d_ft = S3D(num_class=400)
s3d_ft.replace_logits(None)
model_ft = S3DRecognizer(s3d_ft, num_classes=1085)

ckpt_ft = torch.load("checkpoints/finetuning_s3d/s3d_branch3_epoch25.pt", map_location=device)
model_ft.load_state_dict(ckpt_ft["model_state_dict"])

# === Sanity Check: Fine-tuned backbone weights ===
with torch.no_grad():
    w_ft = next(iter(model_ft.s3d.state_dict().values()))
    print(f"=== Fine-tuned S3D Backbone ===")
    print(f"mean: {w_ft.mean().item():.6f}, std: {w_ft.std().item():.6f}, min: {w_ft.min().item():.6f}, max: {w_ft.max().item():.6f}")

# === Compare only the S3D backbone ===
pretrained_params = dict(s3d_pre.named_parameters())
finetuned_params = dict(model_ft.s3d.named_parameters())

for name in pretrained_params:
    if name not in finetuned_params:
        print(f"Missing in fine-tuned model: {name}")
        continue
    p1, p2 = pretrained_params[name], finetuned_params[name]

    if not torch.allclose(p1, p2, atol=1e-6):
        print(f"Parameter changed: {name}")
    else:
        print(f"Match: {name}")
