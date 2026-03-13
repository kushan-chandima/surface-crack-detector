import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras

# Path to your old model
OLD_MODEL_PATH = "models/best_model.keras"
# Path to save the new model
NEW_MODEL_PATH = "models/best_model_resaved.keras"

def main():
    print("Loading old model from:", OLD_MODEL_PATH)
    model = keras.models.load_model(OLD_MODEL_PATH)
    print("Model loaded. Saving to:", NEW_MODEL_PATH)
    model.save(NEW_MODEL_PATH)
    print("Model re-saved successfully.")

if __name__ == "__main__":
    main()
