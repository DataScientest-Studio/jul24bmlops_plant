from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    http_bearer_token: str
    access_token: str
    token_type: str

# use this for create and update
class RoleBase(BaseModel):
    role_name: str
    role_description: Optional[str] = None

class RoleResponse(RoleBase):
    role_id: int
    
    model_config = ConfigDict(from_attributes=True)

    # class Config:
    #     from_attributes = True


# class UserBase(BaseModel):
#     username: str = Field(..., max_length=255)
#     hashed_password: str = Field(..., max_length=255)
#     email: Optional[EmailStr] = None
#     disabled: Optional[bool] = False
#     role_id: int

class UserBase(BaseModel):
    username: str
    hashed_password: str
    email: Optional[EmailStr] = None
    disabled: Optional[bool] = False
    role_id: int

    model_config = ConfigDict(from_attributes=True)

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserUpdate(UserBase):
    password: Optional[str] = Field(None, min_length=6)


class UserResponse(UserBase):
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)

    # class Config:
    #     from_attributes = True
    