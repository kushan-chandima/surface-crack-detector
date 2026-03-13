"""
Model architectures for Surface Crack Detection.
Provides two model options:
  1. MobileNetV2 with Transfer Learning (recommended — ~99% accuracy)
  2. Custom CNN (simpler — ~97% accuracy, similar to Kaggle notebook)
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from src import config


def build_mobilenetv2_model():
    """
    Build a MobileNetV2 transfer learning model for binary classification.

    Architecture:
        - MobileNetV2 base (pre-trained on ImageNet, frozen)
        - GlobalAveragePooling2D
        - Dropout
        - Dense(128, relu)
        - Dropout
        - Dense(1, sigmoid)

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=config.IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the base model
    base_model.trainable = not config.FREEZE_BASE

    # Build classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.DENSE_UNITS, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(1, activation="sigmoid"),
    ], name="MobileNetV2_CrackDetector")

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_custom_cnn_model():
    """
    Build a custom CNN model for binary classification.

    Architecture (similar to referenced Kaggle notebook):
        - 3x [Conv2D → BatchNorm → ReLU → MaxPool2D]
        - GlobalAveragePooling2D
        - Dropout
        - Dense(128, relu)
        - Dense(1, sigmoid)

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    model = models.Sequential(name="CustomCNN_CrackDetector")

    # First conv block
    model.add(layers.Conv2D(config.CNN_FILTERS[0], (3, 3), padding="same",
                            input_shape=config.IMG_SHAPE))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second conv block
    model.add(layers.Conv2D(config.CNN_FILTERS[1], (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third conv block
    model.add(layers.Conv2D(config.CNN_FILTERS[2], (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(config.DROPOUT_RATE))
    model.add(layers.Dense(config.DENSE_UNITS, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_model(model_type=None):
    """
    Factory function to build a model by name.

    Args:
        model_type: "mobilenetv2" or "custom_cnn". Defaults to config.MODEL_TYPE.

    Returns:
        tf.keras.Model: Compiled model.
    """
    model_type = model_type or config.MODEL_TYPE

    builders = {
        "mobilenetv2": build_mobilenetv2_model,
        "custom_cnn": build_custom_cnn_model,
    }

    if model_type not in builders:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: {list(builders.keys())}"
        )

    print(f"\n🏗️  Building model: {model_type}")
    model = builders[model_type]()
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum(
        layer.count_params() for layer in model.layers if layer.trainable
    )
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Non-trainable parameters: {model.count_params() - trainable:,}")

    return model


def load_trained_model(model_path=None):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model. Defaults to config.BEST_MODEL_PATH.

    Returns:
        tf.keras.Model: Loaded model ready for inference.
    """
    from tensorflow.keras.models import load_model

    model_path = model_path or config.BEST_MODEL_PATH

    if not __import__("os").path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Please train the model first with: python scripts/run_training.py"
        )

    print(f"📦 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")

    return model
