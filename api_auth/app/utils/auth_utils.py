from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..schemas.auth_schema import Token, UserBase
from ..config import settings
from ..database.db import get_db
from ..database.tables import User, Role


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_password_hash(password):
     try:
          return pwd_context.hash(password)
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to hash password: {str(e)}")

# Authenticate user function
def authenticate_user(db: Session, username: str, password: str):
     try:
          user = db.query(User).filter(User.username == username).first()
          if not user:
               return False
          if not pwd_context.verify(password, user.hashed_password):
               return False
          return user
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to authenticate user: {str(e)}")

# JWT token creation utility
def create_access_token(data: dict, expires_delta: timedelta = None):
     try:
          to_encode = data.copy()
          if expires_delta:
               expire = datetime.utcnow() + expires_delta
          else:
               expire = datetime.utcnow() + timedelta(minutes=120)
          to_encode.update({"exp": expire})
          encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
          return encoded_jwt
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to create access token: {str(e)}")

# Utility function to verify the token and return the current user
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
     try:
          credentials_exception = HTTPException(
               status_code=status.HTTP_401_UNAUTHORIZED,
               detail="Could not validate credentials",
               headers={"WWW-Authenticate": "Bearer"},
          )
          try:
               payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
               username: str = payload.get("sub")
               if username is None:
                    raise credentials_exception
          except JWTError:
               raise credentials_exception
          user = db.query(User).filter(User.username == username).first()
          if user.disabled:
               raise HTTPException(status_code=400, detail="Inactive user")
          if user is None:
               raise credentials_exception
          return user
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve user: {str(e)}")

# Dependency to check if the current user is an admin
def get_current_admin_user(current_user: UserBase = Depends(get_current_user)):
     print('value of current_user.role.role_name')
     print(current_user.role.role_name)
     try:
          if str(current_user.role.role_name).lower() != 'admin':
               raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
          print('we made it till here')
          return current_user
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to verify admin user: {str(e)}")

