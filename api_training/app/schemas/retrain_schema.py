from pydantic import BaseModel, Field
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
    the_model_id: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# use this for create and update
class ABTestingResultBase(BaseModel):
    test_name: str
    model_a_id: int
    model_b_id: int
    metric_name: str
    model_a_metric_value: float
    model_b_metric_value: float
    winning_the_model_id: Optional[int] = None


class ABTestingResultResponse(ABTestingResultBase):
    test_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class RetrainResponse(BaseModel):
    message: str

    class Config:
        from_attributes = True