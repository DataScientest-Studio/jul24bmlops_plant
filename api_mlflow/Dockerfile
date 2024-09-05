# Stage 1: Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV POETRY_VERSION=1.5.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set the working directory in the build stage
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to install dependencies
COPY pyproject.toml ./

# Install dependencies using Poetry, excluding dev dependencies
RUN poetry install --no-dev --no-root --no-interaction --no-ansi

# Copy the application source code
COPY model/ ./model/
COPY test/ ./test/

# Run tests to ensure the package works correctly
RUN poetry run pytest

# Build the package using Poetry
RUN poetry build


# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

# Set environment variables for the runtime stage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory in the runtime stage
WORKDIR /app

# Copy the built package from the builder stage
COPY --from=builder /app/dist/pr_model-0.1.0-py3-none-any.whl /app/

# Install the package in the runtime environment
RUN pip install /app/pr_model-0.1.0-py3-none-any.whl

# Copy the application source code (if needed)
COPY --from=builder /app/model/mlflow_train.py /app/

# Expose ports for MLFlow and FastAPI
EXPOSE 5000 8000

# Command to run both MLFlow and FastAPI
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & uvicorn mlflow_train:app --host 0.0.0.0 --port 8000"]

