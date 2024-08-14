from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta

from app.schemas.auth import Token, UserBase, UserResponse
from app.config import settings

router = APIRouter()

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
     print('inside login_for_access_token')
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


@app.post("/signup/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
     db_user = db.query(User).filter(User.username == user.username).first()
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

# Read User (Any authenticated user can access their own data)
@app.get("/users/me", response_model=UserResponse)
def read_user_me(current_user: User = Depends(get_current_user)):
     return current_user

# Admin-only Endpoints (CRUD operations for users):
# ==================
@app.get("/users/", response_model=List[UserResponse], dependencies=[Depends(get_current_admin_user)])
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
     users = db.query(User).offset(skip).limit(limit).all()
     return users

@app.get("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(get_current_admin_user)])
def read_user(user_id: int, db: Session = Depends(get_db)):
     db_user = db.query(User).filter(User.user_id == user_id).first()
     if db_user is None:
          raise HTTPException(status_code=404, detail="User not found")
     return db_user

@app.put("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(get_current_admin_user)])
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
     db_user = db.query(User).filter(User.user_id == user_id).first()
     if db_user is None:
          raise HTTPException(status_code=404, detail="User not found")
     for key, value in user.dict(exclude_unset=True).items():
          setattr(db_user, key, value)
     db.commit()
     db.refresh(db_user)
     return db_user

@app.delete("/users/{user_id}", response_model=JSONResponse, dependencies=[Depends(get_current_admin_user)])
def delete_user(user_id: int, db: Session = Depends(get_db)):
     db_user = db.query(User).filter(User.user_id == user_id).first()
     if db_user is None:
          raise HTTPException(status_code=404, detail="User not found")
     db.delete(db_user)
     db.commit()
     return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)


# Role Endpoints (Admin-only access)
# ================================
@app.post("/roles/", response_model=RoleResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_admin_user)])
def create_role(role: RoleCreate, db: Session = Depends(get_db)):
     db_role = db.query(Role).filter(Role.role_name == role.role_name).first()
     if db_role:
          raise HTTPException(status_code=400, detail="Role already exists")
     db_role = Role(role_name=role.role_name, description=role.description)
     db.add(db_role)
     db.commit()
     db.refresh(db_role)
     return db_role

@app.get("/roles/{role_id}", response_model=RoleResponse, dependencies=[Depends(get_current_admin_user)])
def read_role(role_id: int, db: Session = Depends(get_db)):
     db_role = db.query(Role).filter(Role.role_id == role_id).first()
     if db_role is None:
          raise HTTPException(status_code=404, detail="Role not found")
     return db_role

@app.put("/roles/{role_id}", response_model=RoleResponse, dependencies=[Depends(get_current_admin_user)])
def update_role(role_id: int, role: RoleUpdate, db: Session = Depends(get_db)):
     db_role = db.query(Role).filter(Role.role_id == role_id).first()
     if db_role is None:
          raise HTTPException(status_code=404, detail="Role not found")
     for key, value in role.dict(exclude_unset=True).items():
          setattr(db_role, key, value)
     db.commit()
     db.refresh(db_role)
     return db_role

@app.delete("/roles/{role_id}", response_model=JSONResponse, dependencies=[Depends(get_current_admin_user)])
def delete_role(role_id: int, db: Session = Depends(get_db)):
     db_role = db.query(Role).filter(Role.role_id == role_id).first()
     if db_role is None:
          raise HTTPException(status_code=404, detail="Role not found")
     db.delete(db_role)
     db.commit()
     return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)

