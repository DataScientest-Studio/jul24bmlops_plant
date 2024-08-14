from fastapi import APIRouter, UploadFile, File, Depends
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import load_img, img_to_array
from app.utils.data_loader import save_image, ClassLabels
import numpy as np
from io import BytesIO

from app.schemas.model_metadata_schema import ModelMetadataBase, ModelMetadataResponse
from app.schemas.auth_schema import UserBase
from app.database.db import get_db
from app.auth.auth_utils import get_current_admin_user, get_current_user


# Create ModelMetadataBase (Admin-only):
@app.post("/models/", response_model=ModelMetadataResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_admin_user)])
def create_model_metadata(model: ModelMetadataBase, db: Session = Depends(get_db)):
     db_model = ModelMetadataBase(
          model_name=model.model_name,
          version=model.version,
          training_data=model.training_data,
          training_start_time=model.training_start_time,
          training_end_time=model.training_end_time,
          accuracy=model.accuracy,
          f1_score=model.f1_score,
          precision=model.precision,
          recall=model.recall,
          training_loss=model.training_loss,
          validation_loss=model.validation_loss,
          training_accuracy=model.training_accuracy,
          validation_accuracy=model.validation_accuracy,
          training_params=model.training_params,
          logs=model.logs
     )
     db.add(db_model)
     db.commit()
     db.refresh(db_model)
     return db_model


# Read ModelMetadataBase (Any authenticated user):
@app.get("/models/{model_id}", response_model=ModelMetadataResponse)
def read_model_metadata(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
     db_model = db.query(ModelMetadataBase).filter(ModelMetadataBase.model_id == model_id).first()
     if db_model is None:
          raise HTTPException(status_code=404, detail="Model not found")
     return db_model

# Update ModelMetadataBase (Admin-only):
@app.put("/models/{model_id}", response_model=ModelMetadataResponse, dependencies=[Depends(get_current_admin_user)])
def update_model_metadata(model_id: int, model: ModelMetadataBase, db: Session = Depends(get_db)):
     db_model = db.query(ModelMetadataBase).filter(ModelMetadataBase.model_id == model_id).first()
     if db_model is None:
          raise HTTPException(status_code=404, detail="Model not found")
     for key, value in model.dict(exclude_unset=True).items():
          setattr(db_model, key, value)
     db.commit()
     db.refresh(db_model)
     return db_model

# Delete ModelMetadataBase (Admin-only):
@app.delete("/models/{model_id}", response_model=JSONResponse, dependencies=[Depends(get_current_admin_user)])
def delete_model_metadata(model_id: int, db: Session = Depends(get_db)):
     db_model = db.query(ModelMetadataBase).filter(ModelMetadataBase.model_id == model_id).first()
     if db_model is None:
          raise HTTPException(status_code=404, detail="Model not found")
     db.delete(db_model)
     db.commit()
     return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)

