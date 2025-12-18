# Dockerfile.dev
# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONPATH="/app"
WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
# Sync only inference group to .venv
RUN uv sync --frozen --no-install-project

# Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY sentiment_app ./sentiment_app

# Copy model artifacts (ONNX + tokenizer.json + classifier.joblib)
COPY artifacts ./artifacts

COPY src ./src
COPY tokenizer ./tokenizer
COPY *.onnx .

# Run the application
# CMD ["uvicorn", "sentiment_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["app.handler"]