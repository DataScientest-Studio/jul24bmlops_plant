from fastapi import FastAPI
from .endpoints.authentication import router as auth_router
from .endpoints.api_request_log import router as api_request_log_router
from .endpoints.error_log import router as error_log_router
from .database.db import Base, engine
from .config import settings  
import os

app = FastAPI()

# Conditionally initialize the database
# if not settings.TESTING:
# Base.metadata.create_all(bind=engine)

# # Conditionally initialize the database
if os.getenv("TESTING") != "true":
    Base.metadata.create_all(bind=engine)

app.include_router(auth_router, tags=["Authentication"])
app.include_router(api_request_log_router, tags=["API Request Log"])
app.include_router(error_log_router, tags=["Error Log"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Auth API!"}

@app.get("/test/")
def the_test():
    return {"message": "Welcome to the Auth API!"}