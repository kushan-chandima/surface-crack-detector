"""
Streamlit Web App for Surface Crack Detection.
Provides a user-friendly interface for uploading images and getting predictions.

Usage:
    streamlit run app/app.py
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
# Input Selection
# ============================================================================

# =====================
# Input Method Selection
# =====================
input_method = st.radio(
    "Select Input Method:",
    ("📁 Upload Image", "📸 Take a Picture", "🎥 Live Video Stream"),
    horizontal=True,
)

if input_method == "🎥 Live Video Stream":
    import av
    import cv2
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    from src.predict import load_trained_model, _generate_gradcam_heatmap

    show_gradcam_live = st.checkbox("Show Grad-CAM Heatmap (Live)", value=True)
    model = load_model(model_path)
    if model is None:
        st.error(f"❌ Model not found at: `{model_path}`\n\nPlease train the model first.")
    else:
        class CrackDetectionProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.show_gradcam = show_gradcam_live
                self.confidence_threshold = confidence_threshold
                self.frame_count = 0
                self.skip_rate = 2  # Only process every 2nd frame
                self.last_overlay = None
                self.gradcam_skip = 4  # Only compute Grad-CAM every 4th processed frame
                self.last_heatmap = None

            def recv(self, frame):
                self.frame_count += 1
                if self.frame_count % self.skip_rate != 0 and self.last_overlay is not None:
                    # Return last overlay for skipped frames
                    return av.VideoFrame.from_ndarray(self.last_overlay, format="bgr24")

                img = frame.to_ndarray(format="bgr24")
                img_resized = cv2.resize(img, config.IMG_SHAPE[:2][::-1])
                img_array = np.expand_dims(img_resized / 255.0, axis=0)
                probability = float(self.model.predict(img_array, verbose=0)[0][0])
                has_crack = probability >= self.confidence_threshold
                label = "CRACK DETECTED" if has_crack else "NO CRACK"
                color = (0, 0, 255) if has_crack else (0, 255, 0)
                overlay = img.copy()
                # Grad-CAM only every gradcam_skip frames
                if self.show_gradcam:
                    if self.frame_count % (self.skip_rate * self.gradcam_skip) == 0 or self.last_heatmap is None:
                        heatmap = _generate_gradcam_heatmap(self.model, img_array)
                        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                        heatmap = np.uint8(255 * heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        self.last_heatmap = heatmap
                    overlay = cv2.addWeighted(img, 0.7, self.last_heatmap, 0.3, 0)

                # --- Canny edge overlay for visual crack effect ---
                edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                # Overlay edges in green
                overlay = cv2.addWeighted(overlay, 0.9, edges_colored, 0.7, 0)

                cv2.putText(overlay, f"{label} ({probability:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                self.last_overlay = overlay
                return av.VideoFrame.from_ndarray(overlay, format="bgr24")

        webrtc_streamer(
            key="crack-detect",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=CrackDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
else:
    image_file = None
    if input_method == "📁 Upload Image":
        image_file = st.file_uploader(
            "Choose a surface image...",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image of a concrete/surface to check for cracks",
        )
    else:
        image_file = st.camera_input("Take a picture of a surface")

    # Demo mode with sample info
    if image_file is None:
        if input_method == "📁 Upload Image":
            st.info(
                "👆 Upload an image to get started!\n\n"
                "**Supported formats:** JPG, JPEG, PNG, BMP\n\n"
                "**Tip:** Try images of concrete walls, pavements, or building surfaces."
            )
        else:
            st.info(
                "👆 Grant camera access and snap a photo of a surface to check for cracks!"
            )

    if image_file is not None:
        # Display image
        image = Image.open(image_file)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

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
