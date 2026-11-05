FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY api ./api
COPY benchmark ./benchmark
COPY core ./core
COPY generate_benchmark.py ./
COPY results ./results

RUN pip install --no-cache-dir ".[api,agents,benchmark]"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
