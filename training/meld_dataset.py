from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from sklearn.utils.class_weight import compute_class_weight

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir, max_samples=None, apply_augmentation=False):
        print(f"Attempting to load CSV from: {csv_path}")
        print(f"Attempting to load videos from: {video_dir}")
        self.data = pd.read_csv(csv_path)
        if max_samples is not None:
            self.data = self.data.head(max_samples)
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }
        self.apply_augmentation = apply_augmentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if apply_augmentation else None

        # Compute class weights
        self.emotion_labels = [self.emotion_map[label.lower()] for label in self.data['Emotion']]
        self.sentiment_labels = [self.sentiment_map[label.lower()] for label in self.data['Sentiment']]
        emotion_class_weights = compute_class_weight('balanced', classes=np.unique(self.emotion_labels), y=self.emotion_labels)
        sentiment_class_weights = compute_class_weight('balanced', classes=np.unique(self.sentiment_labels), y=self.sentiment_labels)

        # Adjust weights for minority classes (e.g., multiply by 1.5 for rare classes)
        emotion_counts = pd.Series(self.emotion_labels).value_counts(normalize=True)
        sentiment_counts = pd.Series(self.sentiment_labels).value_counts(normalize=True)
        for i in range(len(emotion_class_weights)):
            if emotion_counts.get(i, 0) < 0.1:  # If class frequency is less than 10%
                emotion_class_weights[i] *= 1.5
        for i in range(len(sentiment_class_weights)):
            if sentiment_counts.get(i, 0) < 0.2:  # If class frequency is less than 20%
                sentiment_class_weights[i] *= 1.5

        self.emotion_class_weights = emotion_class_weights
        self.sentiment_class_weights = sentiment_class_weights

    def get_class_weights(self):
        return torch.tensor(self.emotion_class_weights, dtype=torch.float), torch.tensor(self.sentiment_class_weights, dtype=torch.float)

    def _load_video_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found or could not be opened: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"No frames available in video: {video_path}")
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"No frame available in video: {video_path}")

            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1)

            if self.apply_augmentation and self.transform:
                frame_tensor = self.transform(frame_tensor)

            return frame_tensor

        except Exception as e:
            raise ValueError(f"Video frame loading error for {video_path}: {str(e)}")
        finally:
            cap.release()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        path = os.path.join(self.video_dir, video_filename)
        print(f"Processing index {idx}, file: {path}")

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")

            text_inputs = self.tokenizer(
                row['Utterance'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            video_frame = self._load_video_frame(path)

            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frame': video_frame,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing sample at index {idx} (file: {path}): {str(e)}")
            return None

def collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_dataloaders(train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=4, max_samples=None):
    train_dataset = MELDDataset(train_csv, train_video_dir, max_samples, apply_augmentation=True)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir, max_samples)
    test_dataset = MELDDataset(test_csv, test_video_dir, max_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, train_dataset.get_class_weights()

if __name__ == "__main__":
    train_loader, dev_loader, test_loader, (emotion_weights, sentiment_weights) = prepare_dataloaders(
        'dataset/train/train_sent_emo.csv', 'dataset/train/train_splits',
        'dataset/dev/dev_sent_emo.csv', 'dataset/dev/dev_splits_complete',
        'dataset/test/test_sent_emo.csv', 'dataset/test/output_repeated_splits_test',
        max_samples=1000
    )

    print("Emotion class weights:", emotion_weights)
    print("Sentiment class weights:", sentiment_weights)

    for batch in train_loader:
        if batch is None:
            print("No valid batch available, skipping...")
            continue
        print("Text Inputs Sample:", batch['text_inputs'])
        print("Video Frame Shape:", batch['video_frame'].shape)
        print("Emotion Labels:", batch['emotion_label'])
        print("Sentiment Labels:", batch['sentiment_label'])
        break