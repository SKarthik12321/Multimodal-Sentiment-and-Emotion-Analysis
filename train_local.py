import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os

# Load dataset
class SentimentDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data["sentence"]
        self.labels = self.data["emotion"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Set paths
TRAIN_PATH = "./dataset/train/train_sent_emo.csv"
VALID_PATH = "./dataset/dev/dev_sent_emo.csv"
TEST_PATH = "./dataset/test/test_sent_emo.csv"
LOG_DIR = "./training/runs"

# Load data
train_dataset = SentimentDataset(TRAIN_PATH)
valid_dataset = SentimentDataset(VALID_PATH)
test_dataset = SentimentDataset(TEST_PATH)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model
class SentimentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Initialize model
model = SentimentModel(input_dim=100, output_dim=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard writer
writer = SummaryWriter(log_dir=LOG_DIR)

# Training loop
def train_model(epochs=25):
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(torch.randn(len(texts), 100))  # Dummy input features
            loss = criterion(outputs, torch.randint(0, 6, (len(labels),)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Log loss
        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# Run training
if __name__ == "__main__":
    train_model()
    writer.close()