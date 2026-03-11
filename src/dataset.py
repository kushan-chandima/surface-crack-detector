"""
Dataset handling for Surface Crack Detection.
Handles downloading, organizing, splitting, and loading the dataset
with data augmentation and preprocessing.
"""

import os
import shutil
import random
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config


def download_dataset():
    """
    Download the Surface Crack Detection dataset from Kaggle.
    Requires either:
      - Kaggle API credentials (kaggle.json) configured, OR
      - Manual download from: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
    """
    try:
        import opendatasets as od
        print("=" * 60)
        print("Downloading Surface Crack Detection dataset from Kaggle...")
        print("=" * 60)
        print("\nYou will be prompted for your Kaggle credentials.")
        print("Get them from: https://www.kaggle.com/settings -> API -> Create New Token\n")

        od.download(
            f"https://www.kaggle.com/datasets/{config.DATASET_NAME}",
            data_dir=config.DATA_DIR,
        )
        print("\n✅ Dataset downloaded successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nPlease download manually from:")
        print(f"  https://www.kaggle.com/datasets/{config.DATASET_NAME}")
        print(f"\nExtract to: {config.DATA_DIR}/")
        return False


def organize_dataset(source_dir=None, force=False):
    """
    Split the raw dataset into train/val/test directories.

    The raw dataset has structure:
        surface-crack-detection/
            Positive/   (images with cracks)
            Negative/   (images without cracks)

    This function reorganizes into:
        train/Positive/, train/Negative/
        val/Positive/,   val/Negative/
        test/Positive/,  test/Negative/

    Args:
        source_dir: Path to the raw dataset. Defaults to config.RAW_DATA_DIR.
        force: If True, re-split even if split directories already exist.
    """
    source_dir = source_dir or config.RAW_DATA_DIR

    # Check if already organized
    if not force and all(
        os.path.exists(d) and len(os.listdir(d)) > 0
        for d in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]
    ):
        print("✅ Dataset already organized into train/val/test splits.")
        return

    # Validate source directory
    if not os.path.exists(source_dir):
        # Try alternative paths
        alt_paths = [
            os.path.join(config.DATA_DIR, "Surface Crack Detection"),
            os.path.join(config.DATA_DIR, "Concrete Crack Images for Classification"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                source_dir = alt
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found at {source_dir}\n"
                f"Please download from: https://www.kaggle.com/datasets/{config.DATASET_NAME}\n"
                f"And extract to: {config.DATA_DIR}/"
            )

    print(f"\n📂 Organizing dataset from: {source_dir}")
    print(f"   Train split: {config.TRAIN_SPLIT:.0%}")
    print(f"   Val split:   {config.VAL_SPLIT:.0%}")
    print(f"   Test split:  {config.TEST_SPLIT:.0%}\n")

    classes = ["Positive", "Negative"]

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        if not os.path.exists(cls_dir):
            raise FileNotFoundError(f"Class directory not found: {cls_dir}")

        images = sorted(os.listdir(cls_dir))
        images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.seed(42)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * config.TRAIN_SPLIT)
        n_val = int(n_total * config.VAL_SPLIT)

        splits = {
            config.TRAIN_DIR: images[:n_train],
            config.VAL_DIR: images[n_train:n_train + n_val],
            config.TEST_DIR: images[n_train + n_val:],
        }

        for split_dir, split_images in splits.items():
            dest = os.path.join(split_dir, cls)
            os.makedirs(dest, exist_ok=True)

            for img_name in split_images:
                src_path = os.path.join(cls_dir, img_name)
                dst_path = os.path.join(dest, img_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

        print(f"  {cls}: {n_total} images → "
              f"train={n_train}, val={n_total - n_train - len(splits[config.TEST_DIR])}, "
              f"test={len(splits[config.TEST_DIR])}")

    print("\n✅ Dataset organized successfully!")


def create_data_generators():
    """
    Create train, validation, and test data generators with augmentation.

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Training generator WITH augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=config.AUGMENTATION["rotation_range"],
        width_shift_range=config.AUGMENTATION["width_shift_range"],
        height_shift_range=config.AUGMENTATION["height_shift_range"],
        shear_range=config.AUGMENTATION["shear_range"],
        zoom_range=config.AUGMENTATION["zoom_range"],
        horizontal_flip=config.AUGMENTATION["horizontal_flip"],
        vertical_flip=config.AUGMENTATION["vertical_flip"],
        brightness_range=config.AUGMENTATION["brightness_range"],
        fill_mode=config.AUGMENTATION["fill_mode"],
    )

    # Validation & Test generators WITHOUT augmentation (only rescale)
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=True,
        seed=42,
    )

    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    print(f"\n📊 Data Generators Created:")
    print(f"   Train:      {train_generator.samples} images")
    print(f"   Validation: {val_generator.samples} images")
    print(f"   Test:       {test_generator.samples} images")
    print(f"   Image size: {config.IMG_SIZE}")
    print(f"   Batch size: {config.BATCH_SIZE}")

    return train_generator, val_generator, test_generator


def load_single_image(image_path):
    """
    Load and preprocess a single image for prediction.

    Args:
        image_path: Path to the image file.

    Returns:
        numpy array: Preprocessed image with shape (1, 224, 224, 3).
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img = load_img(image_path, target_size=config.IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
