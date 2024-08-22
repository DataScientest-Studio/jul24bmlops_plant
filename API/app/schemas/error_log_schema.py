from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# use this for create
class ErrorLogBase(BaseModel):
     error_type: Optional[str] = None
     error_message: str
     model_id: int
     user_id: int


class ErrorLogResponse(ErrorLogBase):
     error_id: int
     timestamp: datetime

     class Config:
          orm_mode = True