from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_data
from src.modeling.model import get_model
from src.plots import plot_training_history

import os

from sklearn.metrics import accuracy_score

app = typer.Typer()


def train_model(model, train_loader, val_loader, lr=0.001, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return model, history


def test_model(model, test_loader):
    model.eval()

    all_predictions = []
    all_labels = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")


def training_cycle(model_dir=MODELS_DIR):
    train_loader, val_loader, test_loader = get_data()
    model = get_model()

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    model, history1 = train_model(model, train_loader, val_loader, num_epochs=8, lr=0.001)

    for param in model.parameters():
        param.requires_grad = True

    model, history2 = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.0005)

    plot_training_history(history1, history2)

    torch.save(model, os.path.join(model_dir, "fashion_mnist_model_newly_trained.pth"))

    test_model(model, test_loader)

    return model


@app.command()
def main(
        model_path: Path = MODELS_DIR,
        batch_size: int = typer.Option(64, "--batch-size", "-b", help="Training batch size"),
        stage1_epochs: int = typer.Option(8, "--stage1-epochs", help="Epochs for stage 1 (classifier only)"),
        stage2_epochs: int = typer.Option(10, "--stage2-epochs", help="Epochs for stage 2 (fine-tuning)"),
):
    """Train Fashion-MNIST classification model with EfficientNetB0"""
    logger.info(f"Starting training with batch_size={batch_size}, "
                f"stage1_epochs={stage1_epochs}, stage2_epochs={stage2_epochs}")

    training_cycle(
        model_dir=model_path,
        batch_size=batch_size,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs
    )


if __name__ == "__main__":
    app()
