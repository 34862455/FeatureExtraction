import os
import gzip
import pickle
import torch
import torch.nn as nn
import tqdm
from torchvision import models, transforms
from PIL import Image
from extract_features import make_dataset

class FrameFeatureExtractor(nn.Module):
    def __init__(self):
        super(FrameFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)  # (B, 2048)
        return x

class BiLSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2, dropout=0.1):
        super(BiLSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)

    def forward(self, x):  # x: (B, T, D)
        output, _ = self.lstm(x)
        return output  # shape: (B, T, 1024)

def pickle_bilstm_features(feature_root, dataset, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = FrameFeatureExtractor().to(device)
    lstm = BiLSTMFeatureExtractor().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = []
    for files, name, signer, gloss, text in tqdm.tqdm(dataset):
        frame_features = []
        for frame in files:
            image_path = os.path.join(feature_root, name, frame)
            if not os.path.exists(image_path):
                print(f"Missing: {image_path}")
                continue
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
            feat = cnn(img_tensor)
            frame_features.append(feat.squeeze(0).cpu())
        if not frame_features:
            continue
        frame_seq = torch.stack(frame_features).unsqueeze(0).to(device)
        with torch.no_grad():
            features = lstm(frame_seq)
        data.append({
            "name": name,
            "signer": signer,
            "gloss": gloss,
            "text": text,
            "sign": features.squeeze(0).cpu()
        })

    os.makedirs("data", exist_ok=True)
    with gzip.open(f"data/DSG_bilstm_{mode}.pt", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    feature_root_train = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train'
    annotation_file_train = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'

    feature_root_val = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev'
    annotation_file_val = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv'

    feature_root_test = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test'
    annotation_file_test = '/home/minneke/Documents/Dataset/Phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'

    dataset = make_dataset(feature_root_train, annotation_file_train)
    print(f"Train set: {len(dataset)}")
    pickle_bilstm_features(feature_root_train, dataset, "train")

    dataset = make_dataset(feature_root_test, annotation_file_test)
    print(f"Test set: {len(dataset)}")
    pickle_bilstm_features(feature_root_test, dataset, "test")

    dataset = make_dataset(feature_root_val, annotation_file_val)
    print(f"Dev set: {len(dataset)}")
    pickle_bilstm_features(feature_root_val, dataset, "dev")
