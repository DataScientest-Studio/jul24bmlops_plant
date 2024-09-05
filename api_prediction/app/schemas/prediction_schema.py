from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
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
    user_id: Optional[int] = None
    the_model_id: Optional[int] = None
    image_path: Optional[str] = None
    prediction: dict
    top_5_prediction: Optional[List[Dict]] = None
    confidence: float
    feedback_given: Optional[bool] = False
    feedback_comment: Optional[str] = None


class PredictionBaseResponse(PredictionBase):
    prediction_id: int
    predicted_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

    # class Config:
    #     from_attributes = True


