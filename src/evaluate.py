"""
Model evaluation for Surface Crack Detection.
Generates confusion matrix, classification report, ROC curve, and sample predictions.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

from src import config


def evaluate_model(model, test_gen, save_dir=None):
    """
    Full evaluation pipeline: predictions, metrics, and visualizations.

    Args:
        model: Trained Keras model.
        test_gen: Test data generator.
        save_dir: Directory to save evaluation plots. Defaults to config.MODEL_DIR.

    Returns:
        dict: Evaluation metrics (accuracy, AUC, classification report).
    """
    save_dir = save_dir or config.MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📊 Evaluating Model")
    print("=" * 60)

    # Get predictions
    y_pred_proba = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_pred_proba >= config.CONFIDENCE_THRESHOLD).astype(int)
    y_true = test_gen.classes

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n   Test Accuracy: {acc:.4f} ({acc:.2%})")
    print(f"   Test AUC:      {auc:.4f}")

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4,
    )
    print(f"\n📋 Classification Report:\n{report}")

    # Save classification report
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test AUC: {auc:.4f}\n\n")
        f.write(report)
    print(f"   Report saved to: {report_path}")

    # Generate visualizations
    plot_confusion_matrix(y_true, y_pred, save_dir)
    plot_roc_curve(y_true, y_pred_proba, auc, save_dir)
    plot_sample_predictions(model, test_gen, save_dir)

    return {
        "accuracy": acc,
        "auc": auc,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def plot_confusion_matrix(y_true, y_pred, save_dir=None):
    """
    Plot and save a confusion matrix heatmap.
    """
    save_dir = save_dir or config.MODEL_DIR
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_true, y_pred_proba, auc_score, save_dir=None):
    """
    Plot and save the ROC curve with AUC score.
    """
    save_dir = save_dir or config.MODEL_DIR
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
             label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ROC curve saved to: {save_path}")


def plot_sample_predictions(model, test_gen, save_dir=None, n_samples=16):
    """
    Plot a grid of sample predictions from the test set.
    """
    save_dir = save_dir or config.MODEL_DIR

    # Get a batch of images
    test_gen.reset()
    images, labels = next(test_gen)

    # Predict
    predictions = model.predict(images[:n_samples], verbose=0).flatten()

    # Plot grid
    n_cols = 4
    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    for i, ax in enumerate(axes.flat):
        if i >= n_samples or i >= len(images):
            ax.axis("off")
            continue

        ax.imshow(images[i])

        pred_label = config.CLASS_NAMES[int(predictions[i] >= config.CONFIDENCE_THRESHOLD)]
        true_label = config.CLASS_NAMES[int(labels[i])]
        confidence = predictions[i] if predictions[i] >= 0.5 else 1 - predictions[i]
        correct = pred_label == true_label

        color = "green" if correct else "red"
        ax.set_title(
            f"Pred: {pred_label} ({confidence:.1%})\nTrue: {true_label}",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    plt.suptitle("Sample Predictions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Sample predictions saved to: {save_path}")
