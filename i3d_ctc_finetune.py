import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import pandas as pd
from jiwer import wer
from src_i3d.i3dpt import I3D

# ------------------------ Config ------------------------
CHECKPOINT_DIR = "checkpoints/finetuning_i3d"
RESUME_PATH = None
I3D_STRATEGY = "partial"  # Options: "partial", "full"
CHECKPOINT_PREFIX = f"i3d_{I3D_STRATEGY}"
I3D_PRETRAINED = "model_i3d/model_rgb.pth"  # Pretrained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global downsampling factor in temporal dimension after I3D
DOWNSAMPLING_FACTOR = 2  # Set to actual factor after modifying I3D

# ------------------------ Dataset ------------------------
class PhoenixI3DDataset(Dataset):
    def __init__(self, feature_root, annotation_file, vocab, max_frames=200, image_size=(224, 224)):
        self.feature_root = feature_root
        self.vocab = vocab
        self.max_frames = max_frames
        self.image_size = image_size
        self.samples = []

        with open(annotation_file) as f:
            reader = csv.reader(f, delimiter='|')
            next(reader)
            for row in reader:
                name = row[0]
                glosses = row[5].strip().split()
                folder = os.path.join(feature_root, name)
                if not os.path.isdir(folder):
                    continue
                frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')])[:max_frames]
                if len(frames) >= 2:
                    self.samples.append((frames, glosses))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, glosses = self.samples[idx]
        images = []
        for path in frame_paths:
            try:
                img = read_image(path).float() / 255.0
                if img.shape[0] != 3:
                    continue
                img = TF.resize(img, self.image_size)
                img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                images.append(img)
            except:
                continue
        if len(images) < 2:
            raise ValueError(f"Too few frames: {len(images)}")
        clip = torch.stack(images, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        gloss_ids = [self.vocab[g] for g in glosses if g in self.vocab]
        return clip, torch.tensor(gloss_ids, dtype=torch.long), glosses

# ------------------------ Collate ------------------------
def collate_fn(batch):
    clips, gloss_ids, gloss_texts = zip(*batch)
    true_lengths = [clip.shape[1] for clip in clips]  # actual #frames per clip
    max_len = max(true_lengths)

    # Pad clips along the time dimension
    padded_clips = torch.stack([
        F.pad(c, (0, 0, 0, 0, 0, max_len - c.shape[1])) for c in clips
    ])  # (B, C, T, H, W)

    # Flatten targets and store their lengths
    flat_targets = torch.cat(gloss_ids)
    target_lengths = [len(g) for g in gloss_ids]

    return padded_clips, flat_targets, true_lengths, target_lengths, gloss_texts

# ------------------------ Model ------------------------
class I3DRecognizer(nn.Module):
    def __init__(self, i3d_model, feature_dim=1024, num_classes=1066):
        super().__init__()
        self.i3d = i3d_model
        self.classifier = nn.Linear(feature_dim, num_classes + 1)  # +1 for CTC blank token

    def forward(self, x):
        features, _ = self.i3d.extract_features(x)  # (B, C, T, H, W)
        pooled = torch.mean(features, dim=[3, 4])  # (B, C, T)
        pooled = pooled.permute(2, 0, 1)  # (T, B, C)
        logits = self.classifier(pooled)
        return logits

# ------------------------ Decode ------------------------
def greedy_decode(log_probs, idx2gloss):
    pred_ids = log_probs.argmax(2).permute(1, 0)
    results = []
    for seq in pred_ids:
        collapsed, prev = [], -1
        for tok in seq:
            tok = tok.item()
            if tok != prev and tok < len(idx2gloss):
                collapsed.append(tok)
            prev = tok
        results.append(" ".join(idx2gloss[t] for t in collapsed))
    return results

# ------------------------ Vocab ------------------------
def build_vocab(csv_path):
    df = pd.read_csv(csv_path, delimiter='|')
    glosses = sorted({g for row in df.iloc[:, 5] for g in str(row).split()})
    return {g: i for i, g in enumerate(glosses)}, glosses

# ------------------------ Main ------------------------
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Data paths
    train_root = "/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
    dev_root = "/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev"
    train_csv = "/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
    dev_csv = "/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"

    vocab, idx2gloss = build_vocab(train_csv)

    # Load I3D model
    i3d = I3D(num_classes=400)
    weights = torch.load(I3D_PRETRAINED, map_location=device)
    i3d.load_state_dict(weights)
    i3d.to(device).eval()

    # Finetuning strategy
    if I3D_STRATEGY == "partial":
        for name, param in i3d.named_parameters():
            param.requires_grad = "mixed_5c" in name or "mixed_5b" in name
    elif I3D_STRATEGY == "full":
        for param in i3d.parameters():
            param.requires_grad = True

    model = I3DRecognizer(i3d, num_classes=len(vocab)).to(device)

    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters:\n{trainable_params}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CTCLoss(blank=len(vocab), zero_infinity=True)
    scaler = torch.amp.GradScaler()

    # Resume if needed
    start_epoch = 1
    if RESUME_PATH:
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    train_set = PhoenixI3DDataset(train_root, train_csv, vocab)
    val_set = PhoenixI3DDataset(dev_root, dev_csv, vocab)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # ------------------------ Train Loop ------------------------
    for epoch in range(start_epoch, 26):
        model.train()
        total_loss = 0
        for clips, targets, in_lens, tgt_lens, _ in train_loader:
            clips, targets = clips.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                logits = model(clips)  # logits: (T, B, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)  # (T, B, vocab_size)

                # Adjust input lengths based on downsampling
                in_lens_adjusted = torch.div(torch.tensor(in_lens), DOWNSAMPLING_FACTOR, rounding_mode='floor').clamp(min=1).to(device)
                tgt_lens_tensor = torch.tensor(tgt_lens, dtype=torch.long, device=device)

                loss = criterion(log_probs, targets, in_lens_adjusted, tgt_lens_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")

        # ------------------------ Validation ------------------------
        model.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for clips, _, in_lens, _, gloss_txt in val_loader:
                clips = clips.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(clips)
                    log_probs = F.log_softmax(logits, dim=-1)
                    preds = greedy_decode(log_probs, idx2gloss)
                    refs.extend([" ".join(g) for g in gloss_txt])
                    hyps.extend(preds)
        print(f"[Epoch {epoch}] Validation WER: {wer(refs, hyps):.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_epoch{epoch:02d}.pt"))
