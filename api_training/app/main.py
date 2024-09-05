from fastapi import FastAPI
# from .endpoints.retrain import router as retrain_router
from .endpoints.model_metadata import router as model_metadata_router
from .endpoints.ab_testing import router as ab_testing_router
from .database.db import Base, engine

app = FastAPI()

Base.metadata.create_all(bind=engine)

# app.include_router(retrain_router, tags=["Training"])
app.include_router(model_metadata_router, tags=["Model Metadata"])
app.include_router(ab_testing_router, tags=["AB Testing"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Training API!"}