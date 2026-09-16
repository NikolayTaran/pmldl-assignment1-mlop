# Stage 3 — Model API container (FastAPI)
FROM python:3.12-slim

WORKDIR /srv

# copy only what the API needs
COPY code/deployment/api/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api/main.py /srv/main.py

# the model pickle is mounted into the container by docker-compose
# (volume ./models -> /srv/models), so the container always serves
# the model produced by the latest pipeline run
ENV MODEL_DIR=/srv/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
