from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import List
from fastapi.responses import JSONResponse

from app.schemas.auth_schema import Token, UserBase, UserResponse, RoleResponse, RoleBase, UserCreate
from app.config import settings
from app.database.db import get_db
from .auth_utils import get_current_admin_user, get_current_user, authenticate_user, get_password_hash, create_access_token
from app.database.tables import User, Role

router = APIRouter()

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
     try:
          user = authenticate_user(db, form_data.username, form_data.password)
          if not user:
               raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
               )
          
          access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
          access_token = create_access_token(
               data={"sub": user.username}, expires_delta=access_token_expires
          )
          return {"access_token": access_token, "token_type": "bearer"}
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve token: {str(e)}")


@router.post("/signup/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
     try:
          db_user = db.query(User).filter(User.username == user.username).first() or None
          if db_user:
               raise HTTPException(status_code=400, detail="Username already registered")
          hashed_password = get_password_hash(user.password)  
          db_user = User(
               username=user.username,
               email=user.email,
               hashed_password=hashed_password,
               role_id=user.role_id,  # Assume role_id is set during signup
               disabled=user.disabled
          )
          db.add(db_user)
          db.commit()
          db.refresh(db_user)
          return db_user
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to signup: {str(e)}")

# Read User (Any authenticated user can access their own data)
@router.get("/users/me", response_model=UserResponse)
def read_user_me(current_user: UserBase = Depends(get_current_user)):
     try:
          return current_user
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve user info: {str(e)}")

# Admin-only Endpoints (CRUD operations for users):
# ==================
@router.get("/users/", response_model=List[UserResponse], dependencies=[Depends(get_current_admin_user)])
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
     try:
          users = db.query(User).offset(skip).limit(limit).all()
          return users
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve users info: {str(e)}")

@router.get("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(get_current_admin_user)])
def read_user(user_id: int, db: Session = Depends(get_db)):
     try:
          db_user = db.query(User).filter(User.user_id == user_id).first()
          if db_user is None:
               raise HTTPException(status_code=404, detail="User not found")
          return db_user
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve user: {str(e)}")

@router.put("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(get_current_admin_user)])
def update_user(user_id: int, user: UserBase, db: Session = Depends(get_db)):
     try:
          db_user = db.query(User).filter(User.user_id == user_id).first()
          if db_user is None:
               raise HTTPException(status_code=404, detail="User not found")
          for key, value in user.dict(exclude_unset=True).items():
               setattr(db_user, key, value)
          db.commit()
          db.refresh(db_user)
          return db_user
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")

@router.delete("/users/{user_id}", dependencies=[Depends(get_current_admin_user)])
def delete_user(user_id: int, db: Session = Depends(get_db)):
     try:
          db_user = db.query(User).filter(User.user_id == user_id).first()
          if db_user is None:
               raise HTTPException(status_code=404, detail="User not found")
          db.delete(db_user)
          db.commit()
          return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")


# Role Endpoints (Admin-only access)
# ================================
@router.get("/roles/list/", response_model=List[RoleResponse], dependencies=[Depends(get_current_admin_user)])
def list_roles(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
     try:
          roles = db.query(Role).offset(skip).limit(limit).all()
          return roles
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve roles list: {str(e)}")


@router.post("/roles/", response_model=RoleResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_admin_user)])
def create_role(role: RoleBase, db: Session = Depends(get_db)):
     try:
          db_role = db.query(Role).filter(Role.role_name == role.role_name).first()
          if db_role:
               raise HTTPException(status_code=400, detail="Role already exists")
          db_role = Role(role_name=role.role_name, role_description=role.role_description)
          db.add(db_role)
          db.commit()
          db.refresh(db_role)
          return db_role
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to create roles: {str(e)}")

@router.get("/roles/{role_id}", response_model=RoleResponse, dependencies=[Depends(get_current_admin_user)])
def read_role(role_id: int, db: Session = Depends(get_db)):
     try:
          db_role = db.query(Role).filter(Role.role_id == role_id).first()
          if db_role is None:
               raise HTTPException(status_code=404, detail="Role not found")
          return db_role
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve role: {str(e)}")

@router.put("/roles/{role_id}", response_model=RoleResponse, dependencies=[Depends(get_current_admin_user)])
def update_role(role_id: int, role: RoleBase, db: Session = Depends(get_db)):
     try:
          db_role = db.query(Role).filter(Role.role_id == role_id).first()
          if db_role is None:
               raise HTTPException(status_code=404, detail="Role not found")
          for key, value in role.dict(exclude_unset=True).items():
               setattr(db_role, key, value)
          db.commit()
          db.refresh(db_role)
          return db_role
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to update role: {str(e)}")

@router.delete("/roles/{role_id}", dependencies=[Depends(get_current_admin_user)])
def delete_role(role_id: int, db: Session = Depends(get_db)):
     try:
          db_role = db.query(Role).filter(Role.role_id == role_id).first()
          if db_role is None:
               raise HTTPException(status_code=404, detail="Role not found")
          db.delete(db_role)
          db.commit()
          return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to delete role: {str(e)}")

