FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY txncat ./txncat
COPY data ./data

RUN pip install --no-cache-dir . \
    && python -m txncat train --data data/transactions.csv --out models/ml.joblib \
    && useradd --create-home app
USER app

ENV TXNCAT_MODEL=models/ml.joblib \
    TXNCAT_LLM=none
EXPOSE 8000
CMD ["uvicorn", "txncat.api:app", "--host", "0.0.0.0", "--port", "8000"]
