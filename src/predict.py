"""
Prediction module for Surface Crack Detection.
Handles single-image and batch prediction with optional Grad-CAM visualization.
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config
from src.model import load_trained_model
from src.dataset import load_single_image


def predict_image(image_path, model=None, model_path=None):
    """
    Predict whether a surface image contains a crack.

    Args:
        image_path: Path to the input image.
        model: Optional pre-loaded model. If None, loads from model_path.
        model_path: Path to the saved model. Defaults to config.BEST_MODEL_PATH.

    Returns:
        dict: {
            "label": "Positive" or "Negative",
            "confidence": float (0-1),
            "has_crack": bool,
            "probability": float (raw sigmoid output),
        }
    """
    if model is None:
        model = load_trained_model(model_path)

    # Preprocess
    img_array = load_single_image(image_path)

    # Predict
    probability = float(model.predict(img_array, verbose=0)[0][0])
    has_crack = probability >= config.CONFIDENCE_THRESHOLD
    label = "Positive" if has_crack else "Negative"
    confidence = probability if has_crack else 1 - probability

    result = {
        "label": label,
        "confidence": confidence,
        "has_crack": has_crack,
        "probability": probability,
    }

    return result


def predict_with_gradcam(image_path, model=None, model_path=None, save_path=None):
    """
    Predict with Grad-CAM heatmap overlay to visualize model attention.

    Args:
        image_path: Path to the input image.
        model: Optional pre-loaded model.
        model_path: Path to the saved model.
        save_path: Path to save the visualization. If None, shows interactively.

    Returns:
        dict: Prediction result (same as predict_image).
    """
    if model is None:
        model = load_trained_model(model_path)

    # Preprocess
    img_array = load_single_image(image_path)
    original_img = plt.imread(image_path)

    # Get prediction
    result = predict_image(image_path, model=model)

    # Generate Grad-CAM heatmap
    heatmap = _generate_gradcam_heatmap(model, img_array)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Grad-CAM heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Overlay
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    resized = img_to_array(load_img(image_path, target_size=config.IMG_SIZE)) / 255.0
    axes[2].imshow(resized)
    axes[2].imshow(heatmap, cmap="jet", alpha=config.GRAD_CAM_ALPHA)
    axes[2].set_title("Grad-CAM Overlay", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # Add prediction info
    status = "🔴 CRACK DETECTED" if result["has_crack"] else "🟢 NO CRACK"
    fig.suptitle(
        f"{status}  |  Confidence: {result['confidence']:.2%}",
        fontsize=16,
        fontweight="bold",
        color="red" if result["has_crack"] else "green",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📸 Grad-CAM visualization saved to: {save_path}")
    else:
        plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("📸 Visualization saved to: prediction_result.png")

    return result


def _generate_gradcam_heatmap(model, img_array):
    """
    Generate a Grad-CAM heatmap for the given model and image.

    Args:
        model: Trained Keras model.
        img_array: Preprocessed image array (1, H, W, 3).

    Returns:
        numpy array: Heatmap of shape (H, W) normalized to [0, 1].
    """
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            # Handle Sequential models with nested base models
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_layer = sub_layer
                    break
                elif hasattr(sub_layer, 'layers'):
                    for sub_sub_layer in reversed(sub_layer.layers):
                        if isinstance(sub_sub_layer, tf.keras.layers.Conv2D):
                            last_conv_layer = sub_sub_layer
                            break
                    if last_conv_layer:
                        break
            if last_conv_layer:
                break
        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        # Return a blank heatmap if no conv layer found
        return np.zeros(config.IMG_SIZE)

    try:
        # Create a model that outputs both the conv layer output and the predictions
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output],
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv output by the pooled gradients
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize to image size
        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],
            config.IMG_SIZE,
        ).numpy().squeeze()

        return heatmap

    except Exception:
        # Fallback: return blank heatmap
        return np.zeros(config.IMG_SIZE)


def predict_batch(image_paths, model=None, model_path=None):
    """
    Predict crack status for multiple images.

    Args:
        image_paths: List of image file paths.
        model: Optional pre-loaded model.
        model_path: Path to saved model.

    Returns:
        list[dict]: List of prediction results.
    """
    if model is None:
        model = load_trained_model(model_path)

    results = []
    for path in image_paths:
        try:
            result = predict_image(path, model=model)
            result["image_path"] = path
            results.append(result)
        except Exception as e:
            results.append({
                "image_path": path,
                "error": str(e),
            })

    return results


def main():
    """CLI entry point for crack prediction."""
    parser = argparse.ArgumentParser(
        description="Surface Crack Detection — Predict on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Path to the trained model (default: {config.BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--gradcam", "-g",
        action="store_true",
        help="Generate Grad-CAM visualization",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save visualization (default: prediction_result.png)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return

    print(f"\n🔍 Analyzing image: {args.image}")

    if args.gradcam:
        result = predict_with_gradcam(
            args.image,
            model_path=args.model,
            save_path=args.output,
        )
    else:
        result = predict_image(args.image, model_path=args.model)

    # Display result
    print("\n" + "=" * 40)
    status = "🔴 CRACK DETECTED" if result["has_crack"] else "🟢 NO CRACK"
    print(f"  Result:     {status}")
    print(f"  Label:      {result['label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Raw Prob:   {result['probability']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
