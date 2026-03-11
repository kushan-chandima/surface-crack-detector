"""
Unit tests for Surface Crack Detector model module.
Tests model creation, output shapes, and basic prediction behavior.

Usage:
    python -m pytest tests/test_model.py -v
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestModelCreation:
    """Test that models can be instantiated correctly."""

    def test_custom_cnn_creation(self):
        """Test custom CNN model can be created."""
        from src.model import build_model

        model = build_model("custom_cnn")
        assert model is not None
        assert model.name == "CustomCNN_CrackDetector"

    def test_mobilenetv2_creation(self):
        """Test MobileNetV2 model can be created."""
        from src.model import build_model

        model = build_model("mobilenetv2")
        assert model is not None
        assert model.name == "MobileNetV2_CrackDetector"

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        from src.model import build_model

        with pytest.raises(ValueError, match="Unknown model type"):
            build_model("invalid_model")

    def test_factory_default(self):
        """Test factory function uses config default."""
        from src.model import build_model
        from src import config

        model = build_model()
        assert model is not None


class TestModelOutput:
    """Test model output shapes and values."""

    def test_custom_cnn_output_shape(self):
        """Test custom CNN produces correct output shape."""
        from src.model import build_model
        from src import config

        model = build_model("custom_cnn")
        dummy_input = np.random.rand(1, *config.IMG_SHAPE).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        assert output.shape == (1, 1), f"Expected (1,1), got {output.shape}"

    def test_mobilenetv2_output_shape(self):
        """Test MobileNetV2 produces correct output shape."""
        from src.model import build_model
        from src import config

        model = build_model("mobilenetv2")
        dummy_input = np.random.rand(1, *config.IMG_SHAPE).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        assert output.shape == (1, 1), f"Expected (1,1), got {output.shape}"

    def test_output_range(self):
        """Test that model output is between 0 and 1 (sigmoid)."""
        from src.model import build_model
        from src import config

        model = build_model("custom_cnn")
        dummy_input = np.random.rand(5, *config.IMG_SHAPE).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        assert np.all(output >= 0.0), "Output should be >= 0"
        assert np.all(output <= 1.0), "Output should be <= 1"

    def test_batch_prediction(self):
        """Test model handles batch predictions."""
        from src.model import build_model
        from src import config

        model = build_model("custom_cnn")
        batch_size = 8
        dummy_input = np.random.rand(batch_size, *config.IMG_SHAPE).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        assert output.shape == (batch_size, 1)


class TestConfig:
    """Test configuration module."""

    def test_config_values(self):
        """Test that config has required values."""
        from src import config

        assert config.IMG_SIZE == (224, 224)
        assert config.IMG_SHAPE == (224, 224, 3)
        assert config.BATCH_SIZE > 0
        assert config.EPOCHS > 0
        assert config.LEARNING_RATE > 0
        assert len(config.CLASS_NAMES) == 2

    def test_paths_exist_or_creatable(self):
        """Test that config paths reference valid parent directories."""
        from src import config

        assert os.path.isdir(os.path.dirname(config.DATA_DIR))
        assert os.path.isdir(os.path.dirname(config.MODEL_DIR))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
