FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY configs ./configs
COPY scripts ./scripts

EXPOSE 8000

# Default: serve the aggregation API. Override CMD to run the benchmark instead.
CMD ["uvicorn", "oran_fed_robust.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
