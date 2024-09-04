from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


# use this for create
class ErrorLogBase(BaseModel):
     error_type: Optional[str] = None
     error_message: str
     the_model_id: int
     user_id: int


class ErrorLogResponse(ErrorLogBase):
     error_id: int
     timestamp: datetime
     
     model_config = ConfigDict(from_attributes=True)

     # class Config:
     #      from_attributes = True