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
from model_s3d import S3D

# Update:
#   CHECKPOINT_DIR
#   train_root
#   dev_root
#   train_csv
#   dev_csv
#   output path (at bottom)


CHECKPOINT_DIR = '/home/minneke/Documents/Projects/SignExperiments.old/checkpoints/finetuning'
# RESUME_PATH = '/home/minneke/Documents/Projects/SignExperiments.old/checkpoints/finetuning/last.pt'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------ Dataset ------------------------
class PhoenixS3DDataset(Dataset):
    def __init__(self, feature_root, annotation_file, vocab, max_frames=200, image_size=(200, 200)):
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
                if not os.path.isdir(folder): continue
                frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')])[:max_frames]
                if len(frames) >= 2:
                    self.samples.append((frames, glosses))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, glosses = self.samples[idx]
        images = []
        for path in frame_paths:
            try:
                img = read_image(path).float() / 255.0
                if img.shape[0] != 3: continue
                img = TF.resize(img, self.image_size)
                img = img * 2 - 1
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
    lengths = [c.shape[1] for c in clips]
    max_len = max(lengths)
    padded = torch.stack([F.pad(c, (0, 0, 0, 0, 0, max_len - c.shape[1])) for c in clips])  # (B, C, T, H, W)
    flat_targets = torch.cat(gloss_ids)
    target_lengths = [len(g) for g in gloss_ids]
    return padded, flat_targets, lengths, target_lengths, gloss_texts

# ------------------------ Model ------------------------
class S3DRecognizer(nn.Module):
    def __init__(self, s3d_model, feature_dim=1024, num_classes=1066):
        super().__init__()
        self.s3d = s3d_model
        self.classifier = nn.Linear(feature_dim, num_classes + 1)

    def forward(self, x):
        feats = self.s3d(x)           # (B, T, 1024)
        logits = self.classifier(feats)
        return logits.permute(1, 0, 2)  # (T, B, C)

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

if __name__ == "__main__":
    # ------------------------ Setup ------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_root = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train'
    dev_root = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev'
    train_csv = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'
    dev_csv = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv'

    vocab, idx2gloss = build_vocab(train_csv)

    # ------------------------ Load S3D ------------------------
    s3d = S3D(num_class=400)
    weights = torch.load("checkpoints/S3D_kinetics400.pt", map_location=device)
    s3d.load_state_dict({k.replace("module.", ""): v for k, v in weights.items()})
    s3d.replace_logits(None)

    # ------------------------ Load Model ------------------------


    # # ---------------------------------------------last two blocks unfrozen-------------------
    # for name, param in s3d.named_parameters():
    #     if not (name.startswith("base.14") or name.startswith("base.15")):
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    #
    # for name, param in s3d.named_parameters():
    #     if param.requires_grad:
    #         print("Unfrozen:", name)
    #
    # # -----------------------------------------------------------------------------------------

    model = S3DRecognizer(s3d, num_classes=len(vocab)).to(device)
    # ---------------------------------------classifier only------------------------------------
    for param in s3d.parameters():
        param.requires_grad = False #Freeze backbone

    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True #unfreeze classifier only
        else:
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Unfrozen:", name) #manual check

    # --------------------------------------------------------------------------------------------


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CTCLoss(blank=len(vocab), zero_infinity=True)
    scaler = torch.amp.GradScaler()

    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters:\n{trainable_params}")

    start_epoch = 1

    # if RESUME_PATH and os.path.exists(RESUME_PATH):
    #     checkpoint = torch.load(RESUME_PATH, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Resumed from epoch {start_epoch}")


    # ------------------------ Data ------------------------
    train_set = PhoenixS3DDataset(train_root, train_csv, vocab)
    val_set = PhoenixS3DDataset(dev_root, dev_csv, vocab)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # ------------------------ Train Loop ------------------------
    # epoch = start_epoch
    # for epoch in range(start_epoch, 26):
    for epoch in range(1, 26):
        model.train()
        total_loss = 0
        for clips, targets, in_lens, tgt_lens, _ in train_loader:
            clips, targets = clips.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                feats = model.s3d(clips)  # (B, T', 1024)
                log_probs = F.log_softmax(model.classifier(feats), dim=2)  # (B, T', C)
                log_probs = log_probs.permute(1, 0, 2)  # (T', B, C)

                # Adjust input lengths: assume S3D reduces length by 8×
                in_lens_adjusted = [feats.shape[1]] * feats.shape[0]


                loss = criterion(log_probs, targets, in_lens_adjusted, tgt_lens)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for clips, _, in_lens, _, gloss_txt in val_loader:
                clips = clips.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    feats = model.s3d(clips)
                    log_probs = F.log_softmax(model.classifier(feats), dim=2).permute(1, 0, 2)
                    preds = greedy_decode(log_probs, idx2gloss)
                    refs.extend([" ".join(g) for g in gloss_txt])
                    hyps.extend(preds)
        print(f"[Epoch {epoch}] Validation WER: {wer(refs, hyps):.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(CHECKPOINT_DIR, f"s3d_classifier_ft_epoch{epoch:02d}.pt"))
