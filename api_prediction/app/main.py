from fastapi import FastAPI
from .endpoints.prediction import router as prediction_router
from .database.db import Base, engine

app = FastAPI()
Base.metadata.create_all(bind=engine)
app.include_router(prediction_router, tags=["Prediction"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Prediction API!"}