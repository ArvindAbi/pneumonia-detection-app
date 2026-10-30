"""Model loading and preprocessing for the pneumonia API.

The steps here repeat the notebook's prepare_dicom(): decode, grayscale, resize to 512, CLAHE, resize to the model
size, scale to 0 to 1. The notebook checked that these steps give the same scores as the training pipeline.
"""
import io
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pydicom
import keras
from keras import layers

warnings.filterwarnings("ignore", message="The structure of `inputs`")   # harmless Keras notice on every call
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))


@keras.saving.register_keras_serializable(package="pneumo")
class ImageNetCaffePreprocess(layers.Layer):
    """Same layer as in the notebook. It must be defined here so that the saved model can be loaded."""
    MEAN_BGR = (103.939, 116.779, 123.68)

    def call(self, x):
        x = keras.ops.repeat(x * 255.0, 3, axis=-1)
        return x - keras.ops.convert_to_tensor(self.MEAN_BGR, dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 3)


with open(MODEL_DIR / "best_model_config.json") as f:
    CONFIG = json.load(f)
MODEL = keras.models.load_model(MODEL_DIR / "best_model.keras")
CLAHE = cv2.createCLAHE(clipLimit=CONFIG["clahe"]["clip_limit"], tileGridSize=tuple(CONFIG["clahe"]["tile_grid"]))


def decode(data: bytes, filename: str) -> np.ndarray:
    """Returns a uint8 grayscale image from DICOM, PNG or JPG bytes."""
    if filename.lower().endswith(".dcm") or data[128:132] == b"DICM":
        ds = pydicom.dcmread(io.BytesIO(data))
        img = ds.pixel_array
        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("The file could not be read as an image.")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def prepare(img: np.ndarray) -> np.ndarray:
    size, master = CONFIG["img_size"], CONFIG["master_size"]
    img = cv2.resize(img, (master, master), interpolation=cv2.INTER_AREA)
    if CONFIG["enhancement"] == "clahe":
        img = CLAHE.apply(img)
    elif CONFIG["enhancement"] != "control":
        raise ValueError(f"Unsupported enhancement {CONFIG['enhancement']}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img[None, ..., None].astype(np.float32) / 255.0


def predict(data: bytes, filename: str) -> dict:
    x = prepare(decode(data, filename))
    prob = float(MODEL(x, training=False).numpy().ravel()[0])
    thr = float(CONFIG["threshold"])
    return {"predicted_class": "Pneumonia" if prob >= thr else "No pneumonia",
            "probability": round(prob, 4), "threshold": thr, "model": CONFIG["name"]}
