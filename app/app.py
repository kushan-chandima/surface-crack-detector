"""
Streamlit Web App for Surface Crack Detection.
Provides a user-friendly interface for uploading images and getting predictions.

Usage:
    streamlit run app/app.py
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
from PIL import Image
import tempfile

from src import config


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Surface Crack Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .crack-detected {
        background-color: #5e0311;
        border: 2px solid #f44336;
    }
    .no-crack {
        background-color: #023607;
        border: 2px solid #4caf50;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stFileUploader > div > div {
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/search--v1.png", width=60)
    st.title("⚙️ Settings")

    model_path = st.text_input(
        "Model Path",
        value=config.BEST_MODEL_PATH,
        help="Path to the trained model file (.keras)",
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence to classify as crack",
    )

    show_gradcam = st.checkbox(
        "Show Grad-CAM Visualization",
        value=True,
        help="Display heatmap showing where the model focuses",
    )

    st.divider()
    st.markdown("### 📖 About")
    st.markdown("""
    This app uses a **MobileNetV2** deep learning model
    trained on 40,000+ surface images to detect cracks
    with **~99% accuracy**.

    Upload any surface image to check for cracks!
    """)

    st.divider()
    st.markdown(
        "Made with ❤️ using [Streamlit](https://streamlit.io) "
        "& [TensorFlow](https://tensorflow.org)"
    )


# ============================================================================
# Main Content
# ============================================================================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🔍 Surface Crack Detector")
st.markdown("*Upload a surface image to detect cracks using deep learning*")
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# Model Loading (cached)
# ============================================================================
@st.cache_resource
def load_model(path):
    """Load the trained model (cached to avoid reloading)."""
    try:
        from src.model import load_trained_model
        return load_trained_model(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ============================================================================
# File Upload
# ============================================================================
uploaded_file = st.file_uploader(
    "Choose a surface image...",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload an image of a concrete/surface to check for cracks",
)

# Demo mode with sample info
if uploaded_file is None:
    st.info(
        "👆 Upload an image to get started!\n\n"
        "**Supported formats:** JPG, JPEG, PNG, BMP\n\n"
        "**Tip:** Try images of concrete walls, pavements, or building surfaces."
    )

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load model
    model = load_model(model_path)

    if model is None:
        st.error(
            f"❌ Model not found at: `{model_path}`\n\n"
            "Please train the model first:\n"
            "```bash\n"
            "python scripts/run_training.py\n"
            "```"
        )
    else:
        # Save temp file for prediction
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Run prediction
            with st.spinner("🔍 Analyzing image..."):
                from src.predict import predict_image, predict_with_gradcam

                result = predict_image(tmp_path, model=model)

            # Display prediction result
            with col2:
                if result["has_crack"]:
                    st.markdown(
                        '<div class="result-box crack-detected">'
                        '<h2>🔴 CRACK DETECTED</h2>'
                        f'<p class="confidence-text">Confidence: {result["confidence"]:.1%}</p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-box no-crack">'
                        '<h2>🟢 NO CRACK</h2>'
                        f'<p class="confidence-text">Confidence: {result["confidence"]:.1%}</p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                # Detailed metrics
                st.metric("Classification", result["label"])
                st.metric("Raw Probability", f"{result['probability']:.4f}")

            # Grad-CAM visualization
            if show_gradcam:
                st.divider()
                st.subheader("🌡️ Grad-CAM Visualization")
                st.caption("Highlights regions the model focuses on for its prediction")

                gradcam_path = os.path.join(tempfile.gettempdir(), "gradcam_result.png")
                predict_with_gradcam(
                    tmp_path,
                    model=model,
                    save_path=gradcam_path,
                )
                if os.path.exists(gradcam_path):
                    st.image(gradcam_path, use_container_width=True)

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
