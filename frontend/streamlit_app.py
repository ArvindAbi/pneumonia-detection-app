"""Streamlit page: upload a chest X-ray, send it to the Flask API, show the predicted class and score."""
import io
import os

import numpy as np
import pydicom
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:5000")

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("Pneumonia detection from chest X-rays")
st.caption("Capstone project by Arvind Chandrabanu. For study and demonstration only, not for diagnosis.")

try:
    info = requests.get(f"{API_URL}/health", timeout=5).json()
    st.success(f"Model service is running: {info['model']} (decision threshold {info['threshold']:.2f})")
except Exception:
    st.error(f"The model service at {API_URL} is not reachable.")
    st.stop()

upload = st.file_uploader("Upload one chest X-ray", type=["dcm", "png", "jpg", "jpeg"])
if upload is not None:
    data = upload.getvalue()
    if upload.name.lower().endswith(".dcm"):
        arr = pydicom.dcmread(io.BytesIO(data)).pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255
        preview = Image.fromarray(arr.astype(np.uint8))
    else:
        preview = Image.open(io.BytesIO(data))
    left, right = st.columns([1, 1])
    left.image(preview, caption=upload.name, use_container_width=True)
    with st.spinner("Running the model..."):
        resp = requests.post(f"{API_URL}/predict", files={"file": (upload.name, data)}, timeout=120)
    if resp.ok:
        r = resp.json()
        right.subheader("Result")
        right.metric("Predicted class", r["predicted_class"])
        right.metric("Pneumonia probability", f"{r['probability']:.2%}")
        right.progress(min(max(r["probability"], 0.0), 1.0))
        right.write(f"Images scoring {r['threshold']:.2f} or more are flagged as pneumonia. "
                    "The score is the model's output between 0 and 1, not a calibrated probability.")
    else:
        st.error(resp.json().get("error", "The model service returned an error."))

st.divider()
st.write("A flagged image should be read by a radiologist first. The model can miss pneumonia, "
         "especially on PA (standing) X-rays, so a negative result does not rule it out.")
