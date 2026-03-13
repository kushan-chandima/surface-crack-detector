import traceback
from tensorflow.keras.models import load_model
from src.model import build_model
from src import config

print("Attempting to load full model...")
try:
    model = load_model(config.BEST_MODEL_PATH)
    print("SUCCESS full load")
except Exception as e:
    print("FAILED full load")
    traceback.print_exc()

print("Attempting to load weights directly...")
try:
    model2 = build_model(config.MODEL_TYPE)
    model2.load_weights(config.BEST_MODEL_PATH)
    print("SUCCESS load_weights")
except Exception as e:
    import sys
    print("FAILED load_weights:")
    traceback.print_exc(file=sys.stdout)
