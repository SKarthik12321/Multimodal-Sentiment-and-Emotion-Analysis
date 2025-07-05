# evaluate.py (Place this in the top-level project directory)
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from training.meld_dataset import prepare_dataloaders
from training.models import MultimodalSentimentModel  # Import your model

def evaluate_model(model_path, test_csv, test_video_dir):
    """
    Evaluates the trained model on the test set and prints/plots metrics.

    Args:
        model_path: Path to the saved model checkpoint (.pt file).
        test_csv: Path to the test CSV file (e.g., test_sent_emo.csv).
        test_video_dir: Path to the directory containing test videos.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = MultimodalSentimentModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare data loaders (using your existing prepare_dataloaders function)
    _, _, test_loader = prepare_dataloaders(
        train_csv="dummy_train.csv",  # Dummy paths, not used in evaluation
        train_video_dir="dummy_train_dir",
        dev_csv="dummy_dev.csv",
        dev_video_dir="dummy_dev_dir",
        test_csv=test_csv,
        test_video_dir=test_video_dir,
        batch_size=16  # You can adjust the batch size
    )

    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Move data to the correct device
            text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)

            outputs = model(text_inputs, video_frames, audio_features)

            # Get predictions
            all_emotion_preds.append(outputs["emotions"].argmax(dim=1).cpu())
            all_emotion_labels.append(emotion_labels.cpu())
            all_sentiment_preds.append(outputs["sentiments"].argmax(dim=1).cpu())
            all_sentiment_labels.append(sentiment_labels.cpu())

    # Concatenate all predictions and labels
    all_emotion_preds = torch.cat(all_emotion_preds).numpy()
    all_emotion_labels = torch.cat(all_emotion_labels).numpy()
    all_sentiment_preds = torch.cat(all_sentiment_preds).numpy()
    all_sentiment_labels = torch.cat(all_sentiment_labels).numpy()

    # Calculate metrics
    emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average='weighted', zero_division=0)
    emotion_recall = recall_score(all_emotion_labels, all_emotion_preds, average='weighted', zero_division=0)
    emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted', zero_division=0)

    sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average='weighted', zero_division=0)
    sentiment_recall = recall_score(all_sentiment_labels, all_sentiment_preds, average='weighted', zero_division=0)
    sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='weighted', zero_division=0)

    print(f"Emotion Accuracy: {emotion_accuracy:.4f}")
    print(f"Emotion Precision: {emotion_precision:.4f}")
    print(f"Emotion Recall: {emotion_recall:.4f}")
    print(f"Emotion F1-score: {emotion_f1:.4f}")
    print("-" * 20)
    print(f"Sentiment Accuracy: {sentiment_accuracy:.4f}")
    print(f"Sentiment Precision: {sentiment_precision:.4f}")
    print(f"Sentiment Recall: {sentiment_recall:.4f}")
    print(f"Sentiment F1-score: {sentiment_f1:.4f}")


    # Confusion Matrix (Emotion)
    emotion_cm = confusion_matrix(all_emotion_labels, all_emotion_preds)
    emotion_labels_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'] # for plot
    df_cm = pd.DataFrame(emotion_cm, index=emotion_labels_names, columns=emotion_labels_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Emotion Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("emotion_confusion_matrix.png")  # Save the plot
    plt.show()


    # Confusion Matrix (Sentiment)
    sentiment_cm = confusion_matrix(all_sentiment_labels, all_sentiment_preds)
    sentiment_labels_names = ['negative', 'neutral', 'positive'] # for plot
    df_cm = pd.DataFrame(sentiment_cm, index=sentiment_labels_names, columns=sentiment_labels_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Sentiment Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("sentiment_confusion_matrix.png")  # Save the plot
    plt.show()


if __name__ == '__main__':
    # Replace with the actual paths to your model, test CSV, and test video directory
    model_path = "path/to/your/model.pt"  #  !!!CHANGE THIS TO YOUR MODEL PATH!!!
    test_csv = "dataset/test/test_sent_emo.csv" # Example path
    test_video_dir = "dataset/test/output_repeated_splits_test"  # Example path
    evaluate_model(model_path, test_csv, test_video_dir)