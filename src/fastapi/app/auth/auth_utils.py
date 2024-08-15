from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.schemas.auth_schema import Token, UserBase
from app.config import settings
from app.database.db.database import get_db


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_password_hash(password):
     return pwd_context.hash(password)

# Authenticate user function
def authenticate_user(db: Session, username: str, password: str):
     user = db.query(UserBase).filter(UserBase.username == username).first()
     if not user:
          return False
     if not pwd_context.verify(password, user.hashed_password):
          return False
     return user

# JWT token creation utility
def create_access_token(data: dict, expires_delta: timedelta = None):
     to_encode = data.copy()
     if expires_delta:
          expire = datetime.utcnow() + expires_delta
     else:
          expire = datetime.utcnow() + timedelta(minutes=15)
     to_encode.update({"exp": expire})
     encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
     return encoded_jwt


# Utility function to verify the token and return the current user
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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
     user = db.query(UserBase).filter(User.username == username).first()
     if user.disabled:
          raise HTTPException(status_code=400, detail="Inactive user")
     if user is None:
          raise credentials_exception
     return user

# Dependency to check if the current user is an admin
def get_current_admin_user(current_user: UserBase = Depends(get_current_user)):
     if current_user.role.role_name != "admin":
          raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
     return current_user


# fake_users_db = {
#      "arif": {
#           "username": "arif",
#           "full_name": "Example User",
#           "email": "user@example.com",
#           "hashed_password": "fakehashedarif123",
#           "disabled": False,
#      }
# }


# def fake_hash_password(password: str):
#      return "fakehashed" + password

