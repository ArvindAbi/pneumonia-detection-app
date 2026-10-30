#!/bin/sh
# Starts the Flask API in the background and the Streamlit page in the foreground (one container, two processes).
set -e
cd /app/backend && gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app &
cd /app/frontend && exec streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
