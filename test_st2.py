import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import sys

try:
    from src.model import load_trained_model
    model = load_trained_model("models/best_model.keras")
    st.write("Model loaded successfully")
except Exception as e:
    st.write(f"Error: {repr(e)}")
    import traceback
    st.text(traceback.format_exc())
st.write("TF version: ", sys.modules.get('tensorflow').__version__ if 'tensorflow' in sys.modules else "Not loaded")
