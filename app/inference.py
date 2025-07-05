import torch
import torch.nn as nn
from transformers import DistilBertModel, AutoTokenizer
from torchvision import models as vision_models
import cv2
import numpy as np

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
        self.backbone = vision_models.resnet18(pretrained=False)
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

def load_model(model_path):
    model = MultimodalSentimentModel()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1)
    return frame_tensor.unsqueeze(0)

def preprocess_text(text, tokenizer):
    if not text:
        text = " "
    text_inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    return {
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask']
    }

def predict_emotion_sentiment(model, frame, text, tokenizer):
    device = torch.device("cpu")
    model = model.to(device)
    frame_tensor = preprocess_frame(frame).to(device)
    text_inputs = preprocess_text(text, tokenizer)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.inference_mode():
        outputs = model(text_inputs, frame_tensor)
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0].cpu().numpy()
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0].cpu().numpy()
    emotion_map = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    emotion_idx = np.argmax(emotion_probs)
    sentiment_idx = np.argmax(sentiment_probs)
    return {
        'emotion': emotion_map[emotion_idx],
        'emotion_prob': float(emotion_probs[emotion_idx]),
        'sentiment': sentiment_map[sentiment_idx],
        'sentiment_prob': float(sentiment_probs[sentiment_idx]),
        'emotion_probs': {emotion_map[i]: float(prob) for i, prob in enumerate(emotion_probs)},
        'sentiment_probs': {sentiment_map[i]: float(prob) for i, prob in enumerate(sentiment_probs)}
    }