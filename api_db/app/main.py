from fastapi import FastAPI
from .endpoints.database import router as db_router
from .database.db import Base, engine

app = FastAPI()
Base.metadata.create_all(bind=engine)
app.include_router(db_router, tags=["Database Management"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Database Management API!"}