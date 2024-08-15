from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime



# use this for create and update
class ABTestingResultBase(BaseModel):
     test_name: str
     model_a_id: int
     model_b_id: int
     metric_name: str
     model_a_metric_value: float
     model_b_metric_value: float
     winning_model_id: Optional[int] = None


class ABTestingResultResponse(ABTestingResultBase):
     test_id: int
     timestamp: datetime

     class Config:
          orm_mode = True