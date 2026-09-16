# Stage 3 — Web application container (Streamlit)
FROM python:3.12-slim

WORKDIR /srv

COPY code/deployment/app/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app/streamlit_app.py /srv/streamlit_app.py

# the app talks to the API container over the shared docker network;
# override with -e API_URL=... if you run the app outside compose
ENV API_URL=http://api:8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
