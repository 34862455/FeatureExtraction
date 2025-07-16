import torch
import torch.nn as nn
from torchvision.io import read_image,read_video
import torch.nn.functional as F
from torchvision import transforms as t
import cv2
import mediapipe as mp
import json

import matplotlib.pyplot as plt
import copy

import numpy as np
import csv
import random
import os
import pickle
import gzip
from tqdm import tqdm

from model_s3d import S3D
from s3d_ctc_finetune import S3DRecognizer, build_vocab
from PIL import Image, ImageDraw

# ------------------------ Config ------------------------
EXTRACTOR = "i3d" # Options: "s3d", "mediapipe", "i3d"
DATASET = "phoenix"  # Options: "phoenix", "sasl", "how2sign"
CHECKPOINT_PATH = "model_i3d/model_rgb.pth"  # Only for S3D
OUTPUT_NAME = "i3d"  # Prefix for saved feature files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLITS = ["train", "dev", "test"]

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def make_dataset(feature_root, annotation_file):
    dataset = []
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:

                files = sorted([f for f in os.listdir(os.path.join(feature_root,row[0])) if f.endswith('.png')])
                text = row[6].lower()
                name = row[0]
                signer = row[4]
                gloss = row[5]
                dataset.append((files,name,signer,gloss,text))
                line_count += 1

    return dataset

def sort_key(x):
    return int((x.split('_')[1]).replace(".png",""))

def make_dataset_SASL(feature_root, annotation_file):
    folders = os.listdir(feature_root)

    annotation_file = list(filter(lambda x: x["file"].replace(".bag","") in folders, annotation_file ))
    # with open("/home/botlhale/Documents/Mokgadi_masters/SASL_new/SASL/vid_annotations.json","r") as file:
    #     annotations = json.load(file)
    dataset = []
    for row in annotation_file:
            files = sorted([f for f in os.listdir(os.path.join(feature_root,row["file"].replace(".bag",""))) if f.endswith('.png')])
            files.sort(key=sort_key)
            text = row["trans"]
            name = row["file"].replace(".bag","")
            signer = ""
            gloss = ""
            dataset.append((files,name,signer,gloss,text))

    return dataset

def make_dataset_vids(feature_root, annotation_file):
    dataset = []
    with open(annotation_file, encoding = 'cp850') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # print(f"Row[3]: {row[3]}")
                expected_filename = row[3].strip() + '.mp4'  # Strip spaces
                # video_path = os.path.join(feature_root, expected_filename)
                # expected = os.path.join(feature_root+row[3]+'.mp4')
                # print(f"video: {video_path}")
                # print(f"joint:{expected}")
                video_path = os.path.join(feature_root, row[3].strip() + '.mp4')
                if os.path.exists(video_path):
                    # print(f"Found video: {video_path}")
                    text = row[6].lower()
                    name = row[3]
                    signer = ""
                    gloss = ""
                    dataset.append((name,signer,gloss,text))
                # else:
                #     print(f"Missing video: {video_path}")

    print(f"Total videos found: {len(dataset)}")
    return dataset

def load_s3d_model(checkpoint_path):
    from model_s3d import S3D
    from s3d_ctc_finetune import S3DRecognizer, build_vocab

    vocab_file = "/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
    vocab, _ = build_vocab(vocab_file)

    s3d = S3D(num_class=400)
    model = S3DRecognizer(s3d, num_classes=len(vocab))

    checkpoint = torch.load(checkpoint_path)

    # Detect whether it's just a state_dict or a full checkpoint
    state_dict = checkpoint if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint else checkpoint["model_state_dict"]

    # Remove classification head if shapes mismatch
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("fc.0.weight") or k.startswith("fc.0.bias"))
    }

    s3d.load_state_dict(filtered_state_dict, strict=False)

    model.to("cuda").eval()
    for _, p in s3d.named_parameters():
        p.requires_grad = False
    return s3d

def load_i3d_model(weights_path, device):
    from src_i3d.i3dpt import I3D
    if weights_path is None:
        raise ValueError("CHECKPOINT_PATH must be set for I3D extractor.")
    model = I3D(num_classes=400)  # Pretrained on Kinetics-400
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to("cuda").eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_dataset_paths(dataset, split):
    base_path = "/home/minneke/Documents/Dataset"

    if dataset == "phoenix":
        feature_root = f"{base_path}/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{split}"
        annotation_file = f"{base_path}/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{split}.corpus.csv"
    elif dataset == "sasl":
        feature_root = f"{base_path}/SASL_Corpus_png_cropped/SASL Corpus png cropped"
        with open(f"{base_path}/SASL_Corpus_png_cropped/final_no_duplicates_text_num.json", "r") as file:
            annotations = json.load(file)
        cut_off = int(len(annotations) * 0.06)
        if split == "train":
            annotation_file = annotations[2*cut_off:]
        elif split == "dev":
            annotation_file = annotations[0:cut_off]
        elif split == "test":
            annotation_file = annotations[cut_off:2*cut_off]
    elif dataset == "how2sign":
        feature_root = f"{base_path}/How2Sign/{split.capitalize()}/raw_videos"
        annotation_file = f"{base_path}/How2Sign/{split.capitalize()}/how2sign_realigned_{split}.csv"
        # print("Sample files in feature_root:")
        # print(sorted(os.listdir(feature_root))[:5])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return feature_root, annotation_file

def load_dataset(dataset, feature_root, annotation_file):
    if dataset == "phoenix":
        return make_dataset(feature_root, annotation_file)
    elif dataset == "sasl":
        return make_dataset_SASL(feature_root, annotation_file)
    elif dataset == "how2sign":
        return make_dataset_vids(feature_root, annotation_file)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

# ------------------------ Feature Extraction ------------------------
def pickle_features_s3d(feature_root, dataset, output_name, split, model, device):
    data = []

    is_video_dataset = isinstance(dataset[0], tuple) and len(dataset[0]) == 4
    is_frame_dataset = isinstance(dataset[0], tuple) and len(dataset[0]) == 5

    for entry in tqdm.tqdm(dataset, desc=f"Extracting {split}"):
        if is_frame_dataset:
            files, name, signer, gloss, text = entry
            frames = []
            for frame_file in files:
                frame_path = os.path.join(feature_root, name, frame_file)
                try:
                    img = read_image(frame_path)
                    img = t.functional.resize(img, [200, 200])
                    img = (img / 255.) * 2 - 1
                    frames.append(img)
                except Exception as e:
                    print(f"Error reading frame {frame_path}: {e}")
            if len(frames) < 2:
                continue
            frames = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)

        elif is_video_dataset:
            name, signer, gloss, text = entry
            video_path = os.path.join(feature_root, name + '.mp4')
            try:
                video_frames, _, _ = read_video(video_path)
                video_frames = video_frames.permute(0, 3, 1, 2)  # (T, H, W, C) → (T, C, H, W)
                frames = video_frames.permute(1, 0, 2, 3)  # (C, T, H, W)
                frames = t.functional.resize(frames, [200, 200])
                frames = (frames / 255.) * 2 - 1
                if frames.shape[1] < 2:
                    continue
                frames = frames.unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error reading {video_path}: {e}")
                continue

        else:
            raise ValueError("Unknown dataset format")

        with torch.no_grad():
            features = model(frames).squeeze(0)

        data.append({
            "name": name,
            "signer": signer,
            "gloss": gloss,
            "text": text,
            "sign": features.cpu()
        })

    out_path = f"data/DSG_{output_name}_{split}.pt"
    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} samples to {out_path}")

def pickle_features_i3d(feature_root, dataset, output_name, split, model, device):
    data = []
    preprocess_frame = t.Compose([
        t.Resize((224, 224)),
        t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for entry in tqdm(dataset, desc=f"Extracting {split} with I3D"):
        files, name, signer, gloss, text = entry
        frames = []
        for frame_file in files:
            frame_path = os.path.join(feature_root, name, frame_file)
            try:
                img = read_image(frame_path).float() / 255.0
                img = preprocess_frame(img)
                frames.append(img)
            except Exception as e:
                print(f"Error reading frame {frame_path}: {e}")
        if len(frames) < 2:
            continue

        frames = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, C, T, H, W)

        with torch.no_grad():
            raw_features = model.extract_features(frames).squeeze(0)  # (C, T, H, W)
            features = torch.mean(raw_features, dim=[2, 3])  # (C, T)
            features = features.permute(1, 0)  # (T, C)

        data.append({
            "name": name,
            "signer": signer,
            "gloss": gloss,
            "text": text,
            "sign": features.cpu()
        })

    out_path = f"data/DSG_{output_name}_{split}.pt"
    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} samples to {out_path}")

def pickle_features_keypoints(feature_root, dataset, output_name, split):
    import gzip, pickle, torch
    from tqdm import tqdm
    import cv2
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    data = []
    for files, name, signer, gloss, text in tqdm(dataset, desc=f"Extracting keypoints {split}"):
        frames = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for frame in files:
                keypoints = []
                image_path = os.path.join(feature_root, name, frame)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue

                image_height, image_width, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                def extract_landmarks(landmarks, count):
                    if landmarks:
                        for lm in landmarks.landmark[:count]:
                            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                    else:
                        keypoints.extend([0, 0, 0, 0] * count)

                extract_landmarks(results.pose_landmarks, 25)
                extract_landmarks(results.face_landmarks, 468)
                extract_landmarks(results.left_hand_landmarks, 21)
                extract_landmarks(results.right_hand_landmarks, 21)

                frames.append(torch.FloatTensor(keypoints).unsqueeze(0))

        keypoint_seq = torch.cat(frames, dim=0)
        data.append({
            "name": name,
            "signer": signer,
            "gloss": gloss,
            "text": text,
            "sign": keypoint_seq
        })

    out_path = f"data/DSG_keypoints_{output_name}_{split}.pt"
    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} keypoint samples to {out_path}")

# ------------------------ Main Execution ------------------------
def main():
    if EXTRACTOR == "s3d":
        model = load_s3d_model(CHECKPOINT_PATH)
    elif EXTRACTOR == "mediapipe":
        model = None  # Not needed
    elif EXTRACTOR == "i3d":
        model = load_i3d_model(CHECKPOINT_PATH, device)
    else:
        raise ValueError(f"Unsupported extractor: {EXTRACTOR}")

    for split in SPLITS:
        feature_root, annotation_file = get_dataset_paths(DATASET, split)
        dataset = load_dataset(DATASET, feature_root, annotation_file)

        if EXTRACTOR == "s3d":
            pickle_features_s3d(feature_root, dataset, OUTPUT_NAME, split, model, device)
        elif EXTRACTOR == "mediapipe":
            pickle_features_keypoints(feature_root, dataset, OUTPUT_NAME, split)
        elif EXTRACTOR == "i3d":
            pickle_features_i3d(feature_root, dataset, OUTPUT_NAME, split, model, device)


if __name__ == "__main__":
    main()