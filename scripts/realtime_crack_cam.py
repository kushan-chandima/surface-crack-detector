import cv2
import numpy as np
import tensorflow as tf
from src.predict import load_trained_model, _generate_gradcam_heatmap, load_single_image, predict_image
import src.config as config

# Load model once
model = load_trained_model(config.BEST_MODEL_PATH)

# Video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame to model input size
    img_resized = cv2.resize(frame, config.IMG_SHAPE[:2][::-1])
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict
    probability = float(model.predict(img_array, verbose=0)[0][0])
    has_crack = probability >= config.CONFIDENCE_THRESHOLD
    label = "Crack" if has_crack else "No Crack"
    color = (0, 0, 255) if has_crack else (0, 255, 0)

    # Grad-CAM heatmap
    heatmap = _generate_gradcam_heatmap(model, img_array)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    # Draw label and confidence
    cv2.putText(overlay, f"{label} ({probability:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Crack Detection (Grad-CAM)', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
