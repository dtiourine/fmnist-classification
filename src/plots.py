from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_training_history(history1, history2):
    train_loss = history1["train_loss"] + history2["train_loss"]
    train_acc = history1["train_accuracy"] + history2["train_accuracy"]
    val_loss = history1["val_loss"] + history2["val_loss"]
    val_acc = history1["val_accuracy"] + history2["val_accuracy"]

    epochs = range(1, len(train_loss) + 1)
    stage1_epochs = len(history1["train_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.axvline(x=stage1_epochs + 0.5, color='gray', linestyle='--', alpha=0.7,
                label='Stage 1 → Stage 2')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.text(stage1_epochs / 2, max(train_loss) * 0.9, 'Stage 1\n(Classifier Only)',
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.text(stage1_epochs + (len(epochs) - stage1_epochs) / 2, max(train_loss) * 0.9,
             'Stage 2\n(Fine-tuning)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axvline(x=stage1_epochs + 0.5, color='gray', linestyle='--', alpha=0.7,
                label='Stage 1 → Stage 2')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.text(stage1_epochs / 2, min(train_acc) + (max(train_acc) - min(train_acc)) * 0.1,
             'Stage 1\n(Classifier Only)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax2.text(stage1_epochs + (len(epochs) - stage1_epochs) / 2,
             min(train_acc) + (max(train_acc) - min(train_acc)) * 0.1,
             'Stage 2\n(Fine-tuning)', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    plt.tight_layout()
    plt.show()

    print(
        f"Stage 1 - Final Train Acc: {history1['train_accuracy'][-1]:.2f}%, Val Acc: {history1['val_accuracy'][-1]:.2f}%")
    print(
        f"Stage 2 - Final Train Acc: {history2['train_accuracy'][-1]:.2f}%, Val Acc: {history2['val_accuracy'][-1]:.2f}%")
    print(f"Best Validation Accuracy: {max(val_acc):.2f}%")
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
