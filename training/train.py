import argparse
import torch
from torch.utils.data import DataLoader
from models import MultimodalSentimentModel, MultimodalTrainer
from meld_dataset import prepare_dataloaders
import os
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)  # Increased to 10
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)  # Reduced to 3e-5
    parser.add_argument("--train-dir", type=str, default="dataset/train")
    parser.add_argument("--val-dir", type=str, default="dataset/dev")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--model-dir", type=str, default="models")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu")

    print("Initial CPU memory stats not available (CPU-only device)")

    train_csv = os.path.join(args.train_dir, "train_sent_emo.csv")
    val_csv = os.path.join(args.val_dir, "dev_sent_emo.csv")
    test_csv = os.path.join(args.test_dir, "test_sent_emo.csv")
    train_loader, val_loader, test_loader, (emotion_weights, sentiment_weights) = prepare_dataloaders(
        train_csv, os.path.join(args.train_dir, "train_splits"),
        val_csv, os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv, os.path.join(args.test_dir, "output_repeated_splits_test"),
        batch_size=args.batch_size,
        max_samples=1000  # Increased to 1000
    )

    print(f"Training CSV path: {train_csv}")
    print(f"Training video directory: {os.path.join(args.train_dir, 'train_splits')}")

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(
        model, 
        train_loader, 
        val_loader, 
        learning_rate=args.learning_rate,
        emotion_class_weights=emotion_weights,
        sentiment_class_weights=sentiment_weights
    )
    best_val_loss = float('inf')

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_losses = trainer.train_epoch()
        val_losses, val_metrics = trainer.evaluate(val_loader)

        metrics_data["train_losses"].append(train_losses["total"])
        metrics_data["val_losses"].append(val_losses["total"])
        metrics_data["epochs"].append(epoch)

        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_losses["total"]},
                {"Name": "validation:loss", "Value": val_losses["total"]},
                {"Name": "validation:emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]},
                {"Name": "validation:emotion_precision", "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:sentiment_precision", "Value": val_metrics["sentiment_precision"]},
            ]
        }))

        print("Memory stats not available (CPU-only device)")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

    print("Evaluating on test set...")
    test_losses, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_losses["total"]

    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_losses["total"]},
            {"Name": "test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:emotion_precision", "Value": test_metrics["emotion_precision"]},
            {"Name": "test:sentiment_precision", "Value": test_metrics["sentiment_precision"]},
        ]
    }))

if __name__ == "__main__":
    main()