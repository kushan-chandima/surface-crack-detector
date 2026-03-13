"""
Training pipeline for Surface Crack Detection.
Handles model training with callbacks, history visualization, and model saving.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src import config


def get_callbacks():
    """
    Create training callbacks for early stopping, model checkpointing,
    learning rate scheduling, and TensorBoard logging.

    Returns:
        list: List of Keras callback instances.
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save the best model during training
        ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=config.LOG_DIR,
            histogram_freq=1,
        ),
    ]

    return callbacks


def train_model(model, train_gen, val_gen, epochs=None):
    """
    Train the model with the given data generators and callbacks.

    Args:
        model: Compiled Keras model.
        train_gen: Training data generator.
        val_gen: Validation data generator.
        epochs: Number of epochs (defaults to config.EPOCHS).

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    epochs = epochs or config.EPOCHS
    callbacks = get_callbacks()

    print("\n" + "=" * 60)
    print("🚀 Starting Training")
    print("=" * 60)
    print(f"   Model:      {model.name}")
    print(f"   Epochs:     {epochs}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   LR:         {config.LEARNING_RATE}")
    print("=" * 60 + "\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.FINAL_MODEL_PATH)
    print(f"\n💾 Final model saved to: {config.FINAL_MODEL_PATH}")
    print(f"💾 Best model saved to:  {config.BEST_MODEL_PATH}")

    return history


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras History object from model.fit().
        save_path: Path to save the plot. If None, saves to models/training_history.png.
    """
    save_path = save_path or os.path.join(config.MODEL_DIR, "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n📈 Training history plot saved to: {save_path}")
