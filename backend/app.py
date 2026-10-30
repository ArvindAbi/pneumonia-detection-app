"""Flask API for the pneumonia classifier.

GET  /health   returns the model name and threshold
POST /predict  takes one X-ray (DICOM, PNG or JPG) as form field "file" and returns the predicted class and score
"""
import os

from flask import Flask, jsonify, request

import inference

app = Flask(__name__)
ALLOWED = (".dcm", ".png", ".jpg", ".jpeg")


@app.get("/health")
def health():
    return jsonify(status="ok", model=inference.CONFIG["name"], threshold=inference.CONFIG["threshold"])


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="Send the X-ray as form field 'file'."), 400
    if not f.filename.lower().endswith(ALLOWED):
        return jsonify(error="Accepted formats: DICOM (.dcm), PNG, JPG."), 400
    try:
        return jsonify(inference.predict(f.read(), f.filename))
    except Exception as err:
        return jsonify(error=f"Could not process the image: {err}"), 422


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
