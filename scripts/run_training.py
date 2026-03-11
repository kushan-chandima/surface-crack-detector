"""
End-to-end training script for Surface Crack Detection.
Downloads data (if needed), trains the model, evaluates, and saves results.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --model mobilenetv2 --epochs 20
    python scripts/run_training.py --model custom_cnn --epochs 30
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.dataset import download_dataset, organize_dataset, create_data_generators
from src.model import build_model
from src.train import train_model, plot_training_history
from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train Surface Crack Detector")
    parser.add_argument(
        "--model", "-m",
        default=config.MODEL_TYPE,
        choices=["mobilenetv2", "custom_cnn"],
        help=f"Model architecture (default: {config.MODEL_TYPE})",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=config.EPOCHS,
        help=f"Number of training epochs (default: {config.EPOCHS})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (assumes data is already available)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🏗️  Surface Crack Detector — Training Pipeline")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch:  {config.BATCH_SIZE}")
    print(f"  LR:     {config.LEARNING_RATE}")
    print("=" * 60)

    # Step 1: Download and organize data
    if not args.skip_download:
        print("\n📥 Step 1/4: Downloading dataset...")
        download_dataset()

    print("\n📂 Step 2/4: Organizing dataset...")
    try:
        organize_dataset()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nRun with --skip-download if data is already organized.")
        return

    # Step 2: Create data generators
    print("\n📊 Step 2/4: Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators()

    # Step 3: Build and train model
    print("\n🏗️  Step 3/4: Building and training model...")
    model = build_model(model_type=args.model)
    model.summary()

    history = train_model(model, train_gen, val_gen, epochs=args.epochs)

    # Plot training history
    plot_training_history(history)

    # Step 4: Evaluate
    print("\n📊 Step 4/4: Evaluating model...")
    results = evaluate_model(model, test_gen)

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print("=" * 60)
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test AUC:      {results['auc']:.4f}")
    print(f"\n  Model saved to: {config.BEST_MODEL_PATH}")
    print(f"  Plots saved to: {config.MODEL_DIR}/")
    print("\n  To predict on a new image:")
    print(f"    python -m src.predict --image <path_to_image>")
    print("\n  To launch the web app:")
    print(f"    streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
