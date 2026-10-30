FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 API_URL=http://localhost:5000
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r frontend/requirements.txt

COPY backend backend
COPY frontend frontend
COPY model model
COPY start.sh start.sh

EXPOSE 5000 8501
CMD ["sh", "start.sh"]
