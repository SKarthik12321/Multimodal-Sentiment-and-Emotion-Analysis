import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision import models as vision_models
from sklearn.metrics import accuracy_score, precision_score

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.projection = nn.Linear(768, 64)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(pooler_output)

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        return self.backbone(x)

class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.emotion_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 7)
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 3)
        )

    def forward(self, text_inputs, video_frame):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(video_frame)
        combined_features = torch.cat([text_features, video_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)
        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }

    @staticmethod
    def load_model(model_path):
        model = MultimodalSentimentModel()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model

class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=2e-5, emotion_class_weights=None, sentiment_class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        device = torch.device("cpu")
        self.emotion_criterion = nn.CrossEntropyLoss(weight=emotion_class_weights.to(device) if emotion_class_weights is not None else None)
        self.sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_class_weights.to(device) if sentiment_class_weights is not None else None)

    def train_epoch(self):
        self.model.train()
        running_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}

        for batch in self.train_loader:
            if batch is None:
                continue
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frame = batch['video_frame'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)

            self.optimizer.zero_grad()
            outputs = self.model(text_inputs, video_frame)
            emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs["sentiments"], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

        return {k: v/len(self.train_loader) for k, v in running_loss.items()}

    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                if batch is None:
                    continue
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frame = batch['video_frame'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

                outputs = self.model(text_inputs, video_frame)
                emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs["sentiments"], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extend(outputs["emotions"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(outputs["sentiments"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

        avg_loss = {k: v/len(data_loader) for k, v in losses.items()}
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average='weighted', zero_division=0)
        sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average='weighted', zero_division=0)

        # Update learning rate scheduler
        if phase == "val":
            self.scheduler.step(avg_loss['total'])

        return avg_loss, {
            'emotion_accuracy': emotion_accuracy,
            'sentiment_accuracy': sentiment_accuracy,
            'emotion_precision': emotion_precision,
            'sentiment_precision': sentiment_precision
        }

if __name__ == "__main__":
    from meld_dataset import MELDDataset
    dataset = MELDDataset('dataset/train/train_sent_emo.csv', 'dataset/train/train_splits', max_samples=1000)
    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frame = sample['video_frame'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frame)
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    emotion_map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Predictions for utterance")