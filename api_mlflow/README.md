# MLOps for Plant Recognition

This project integrates MLFlow for model management and FastAPI for serving a web interface to interact with machine learning models. The Docker container is designed to run both MLFlow and FastAPI concurrently, allowing you to train, log, and serve models efficiently.

## Project Structure

mlops-pr/ ├── Dockerfile ├── pyproject.toml ├── model/ │ ├── mlflow_train.py # Main script handling MLFlow operations │ └── your_model_files.py # Additional model-related files └── test/ └── test_files.py # Unit tests


## Overview

- **MLFlow**: Handles model training, logging, and loading.
- **FastAPI**: Serves as an interface to interact with the MLFlow models via HTTP endpoints.
- **Docker**: Containerizes the entire application, running both MLFlow and FastAPI concurrently.

## Prerequisites

- Docker
- Python 3.11
- Poetry

## Setup

### 1. Install Dependencies

This project uses Poetry to manage dependencies. Install Poetry using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Once Poetry is installed, you can install the project dependencies:

```bash
poetry install --no-dev --no-root --no-interaction --no-ansi
```

### 2. Dockerfile Breakdown

The Dockerfile is divided into two stages: the build stage and the runtime stage.

**Build Stage**
- **Installing Dependencies**: Installs all required dependencies using Poetry.
- **Running Tests**: Ensures the code is working correctly by running unit tests.
- **Building the Package**: Packages the project into a distributable format.

**Runtime Stage**
- **Installing the Built Package**: Installs the package built in the previous stage.
- **Running MLFlow and FastAPI**: Starts both MLFlow and FastAPI services concurrently within the container.

### 3. MLFlow and FastAPI Integration

**MLFlow (`mlflow_train.py`)**

The mlflow_train.py script is responsible for:
- **Training the Model**: Loads data, trains the model, and logs it using MLFlow.
- **Logging the Model**: Logs parameters, metrics, and model artifacts to MLFlow.
- **Loading the Model**: Loads the logged model for inference.

**FastAPI (`app.py`)**

The FastAPI application provides endpoints for:
- **Training the Model (/train)**: Triggers the training and logging process.
- **Making Predictions (/predict)**: Allows you to make predictions using the trained model.
- **Fetching Model Information (/model_info/{model_name})**: Retrieves details about the logged models.

### 4. Running the Application
You can build and run the Docker container using the following commands:

**Build the Docker Image**

```bash
docker build -t mlops-pr .
```

**Run the Docker Container**

```bash
docker run -p 5000:5000 -p 8000:8000 mlops-pr
```

- MLFlow UI: Accessible at http://localhost:5000.
- FastAPI: Accessible at http://localhost:8000.

### 5. Example Endpoints
- Train the Model:
  - POST /train
  - Example Request: curl -X POST http://localhost:8000/train
- Make a Prediction:
  - POST /predict
  - Example Request:
    ```bash
    curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
    ```
- Get Model Information:
  - GET /model_info/{model_name}
  - Example Request: `curl http://localhost:8000/model_info/pr_448`

### 6. Additional Information
- **Dependencies**: All dependencies are managed by Poetry and are specified in the `pyproject.toml` file.
- **Testing**: Unit tests are run during the build stage to ensure code correctness.
- **Versioning**: The project is currently at version `0.1.0`.
