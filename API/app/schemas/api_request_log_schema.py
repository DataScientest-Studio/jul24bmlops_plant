from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# use this for create
class APIRequestLogBase(BaseModel):
     endpoint: str
     request_method: str
     request_body: Optional[str] = None
     response_status: int
     response_time_ms: Optional[float] = None
     user_id: int
     ip_address: Optional[str] = None


class APIRequestLogResponse(APIRequestLogBase):
     request_id: int
     timestamp: datetime

     class Config:
          orm_mode = True
