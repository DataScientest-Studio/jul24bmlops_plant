# Stage 1: Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV POETRY_VERSION=1.5.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies and Poetry in a single RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml, poetry.lock, and README files to install dependencies
COPY pyproject.toml README.md ./

# Install dependencies using Poetry, excluding dev dependencies
RUN poetry install --no-dev --no-root --no-interaction --no-ansi

# Copy the remaining application source code
COPY model/ ./model/
COPY test/ ./test/

# Run tests to ensure the package works correctly in a single step
RUN poetry run pytest && poetry build


# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

# Set environment variables for the runtime stage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACK_DIR_PATH="/app/MLFlow" \
    MLFLOW_MODEL_DIR_PATH="/app/Model" \
    IMAGES_DIR_PATH="/app/Data"

# Set the working directory
WORKDIR /app

# Copy the built package from the builder stage
COPY --from=builder /app/dist/pr_model-0.1.0-py3-none-any.whl /app/model/ /app/

# Create directories and install the package in a single RUN command
RUN mkdir -p /app/MLFlow /app/Model /app/Data && \
    pip install /app/pr_model-0.1.0-py3-none-any.whl

# Expose ports for MLFlow and FastAPI
EXPOSE 5000 8005

# Command to run both MLFlow and FastAPI
CMD ["sh", "-c", "mlflow ui --backend-store-uri ${MLFLOW_TRACK_DIR_PATH} --host 0.0.0.0 --port 5000 & uvicorn app:app --host 0.0.0.0 --port 8005"]
