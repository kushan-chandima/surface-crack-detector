"""
Script to download the Surface Crack Detection dataset from Kaggle.

Usage:
    python scripts/download_data.py

Requirements:
    - opendatasets package (included in requirements.txt)
    - Kaggle API credentials (username and key from kaggle.com/settings)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import download_dataset, organize_dataset
from src import config


def main():
    print("=" * 60)
    print("  Surface Crack Detection — Dataset Download")
    print("=" * 60)
    print(f"\n  Dataset: {config.DATASET_NAME}")
    print(f"  Target:  {config.DATA_DIR}\n")

    # Download
    success = download_dataset()

    if success:
        # Organize into train/val/test
        try:
            organize_dataset()
        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")
            print("\nPlease check the data directory and try again.")
            return

    print("\n" + "=" * 60)
    print("  ✅ Dataset setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
