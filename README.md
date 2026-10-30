# Pneumonia detection from chest X-rays: web app

This is the deployment part of my PGP-AIML capstone project, Computer Vision Pneumonia Detection. You upload one chest X-ray, and the app shows the predicted class (Pneumonia or No pneumonia) and the model's score.

It is for study and demonstration only. It is not a medical device and must not be used for diagnosis.

## How it is built

The browser talks to the Streamlit page (port 8501). The page sends the uploaded file over HTTP to the Flask API (port 5000), and the API runs the Keras model in `model/best_model.keras`.

| Part | Folder | What it does | Dependencies |
|---|---|---|---|
| Backend | `backend/` | Flask API. `GET /health` returns the model name and threshold. `POST /predict` takes one X-ray (DICOM, PNG or JPG, form field `file`) and returns the predicted class, the score and the threshold | `backend/requirements.txt` (TensorFlow CPU, Keras, Flask, Gunicorn, OpenCV, pydicom) |
| Frontend | `frontend/` | Streamlit page: upload, preview and result | `frontend/requirements.txt` (Streamlit, requests, Pillow, pydicom) |
| Model | `model/` | The final model from the notebook, plus `best_model_config.json` (threshold, image size, preprocessing) | |
| Container | `Dockerfile`, `start.sh`, `docker-compose.yml` | One image that runs both processes | |
| Codespaces | `.devcontainer/devcontainer.json` | Docker inside the codespace; ports 8501 and 5000 are forwarded | |

The API prepares each image exactly as in training: it decodes the image to grayscale, resizes it to 512 x 512, applies CLAHE (clip limit 2.0, 8 x 8 tiles), resizes it to the model size, and scales it to 0 to 1. The conversion to the 3-channel ImageNet input happens inside the saved model, in the custom layer `ImageNetCaffePreprocess`. `backend/inference.py` defines the same layer so that the model can be loaded.

## Run in GitHub Codespaces

1. On the repository page, choose **Code > Codespaces > Create codespace on main**.
2. When the terminal is ready, run:
   ```bash
   docker compose up --build
   ```
3. Open the **Ports** tab. The forwarded address for port **8501** (Streamlit app) has the form `https://<codespace-name>-8501.app.github.dev`.
4. Upload a chest X-ray and read the result.

To test the API alone (inside the codespace):

```bash
curl http://localhost:5000/health
curl -F "file=@my_xray.dcm" http://localhost:5000/predict
```

## Run locally without Docker

```bash
pip install -r backend/requirements.txt -r frontend/requirements.txt
cd backend && python app.py
cd frontend && streamlit run streamlit_app.py
```

Run the last two commands in two separate terminals.
