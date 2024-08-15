from pydantic import BaseModel
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

from typing import List

class TopPrediction(BaseModel):
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_5_predictions: List[TopPrediction]
    message: str


# use this for create
class PredictionBase(BaseModel):
    user_id: int
    model_id: int
    image_path: Optional[str] = None
    prediction: dict
    top_5_prediction: Optional[dict] = None
    confidence: float
    feedback_given: Optional[bool] = False
    feedback_comment: Optional[str] = None


class PredictionBaseResponse(PredictionBase):
    prediction_id: int
    predicted_at: datetime

    class Config:
        orm_mode = True