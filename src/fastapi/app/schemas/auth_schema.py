from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str

# use this for create and update
class RoleBase(BaseModel):
    role_name: str
    description: Optional[str] = None

class RoleResponse(RoleBase):
    role_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    username: str = Field(..., max_length=255)
    hashed_password: str = Field(..., max_length=255)
    email: Optional[EmailStr] = None
    disabled: Optional[bool] = False
    role_id: int


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserUpdate(UserBase):
    password: Optional[str] = Field(None, min_length=6)


class UserResponse(UserBase):
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True