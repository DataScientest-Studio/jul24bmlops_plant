from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# use this for create and update
class ModelMetadataBase(BaseModel):
     model_name: str
     version: str
     training_data: str
     training_start_time: datetime
     training_end_time: datetime
     accuracy: float
     f1_score: Optional[float] = None
     precision: Optional[float] = None
     recall: Optional[float] = None
     training_loss: float
     validation_loss: float
     training_accuracy: float
     validation_accuracy: float
     training_params: dict
     logs: str


class ModelMetadataResponse(ModelMetadataBase):
     model_id: int
     created_at: Optional[datetime]
     updated_at: Optional[datetime]

     class Config:
          orm_mode = True