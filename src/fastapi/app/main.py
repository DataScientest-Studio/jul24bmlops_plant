from fastapi import FastAPI, Depends
from app.auth.authentication import router as auth_router
from app.prediction_management.prediction import router as predict_router
from app.prediction_management.retrain import router as retrain_router
from app.prediction_management.collect_data import router as collect_data_router
from app.prediction_management.my_auth import router as ab_router
from app.prediction_management.api_request_log import router as api_request_log_router
from app.prediction_management.error_log import router as error_log_router
from app.prediction_management.model_metadata import router as model_metadata_router

from app.auth.auth_utils import get_current_user

app = FastAPI()

# You need to create the tables in your PostgreSQL database. 
# You can do this using SQLAlchemy directly or with Alembic for migrations.
# Option 1: Directly with SQLAlchemy
# Add the following to main.py:

## from .database import engine
# from app.database.database import Base, engine
# Base.metadata.create_all(bind=engine)

# Using Alembic:
# Steps:
# Initialize Alembic: Run alembic init alembic to set up Alembic in your project.
# Create Migrations: When you make changes to your SQLAlchemy models, 
# you create a new migration script with alembic revision --autogenerate -m "Description of change".
# Apply Migrations: Run alembic upgrade head to apply the migration to your database.
# Manage Versions: Alembic keeps track of schema versions, allowing you to upgrade or downgrade the 
# database schema as needed.


# Include routers for different functionalities
app.include_router(auth_router, tags=["Authentication"])
app.include_router(predict_router, tags=["Prediction"])
app.include_router(retrain_router, tags=["Retrain"])
app.include_router(collect_data_router, tags=["Collecting Data"])
app.include_router(ab_router)
app.include_router(api_request_log_router, tags=["API Request Log"])
app.include_router(error_log_router, tags=["Error Log"])
app.include_router(model_metadata_router, tags=["Model Metadata"])

@app.get("/")
def read_root():
     return {"message": "Welcome to the Plant Detection API!"}

@app.get("/test-api")
def read_root(current_user: str = Depends(get_current_user)):
     return {"message": "The API is up and running with authorization"}
