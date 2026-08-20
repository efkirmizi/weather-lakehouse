FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py dashboard.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Expose ports for both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

CMD ["./entrypoint.sh"]
