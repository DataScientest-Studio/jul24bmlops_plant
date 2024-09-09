from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text, TIMESTAMP, JSON, func, DateTime
from sqlalchemy.orm import relationship
from .db import Base

class User(Base):
     __tablename__ = "users"
     
     user_id = Column(Integer, primary_key=True, index=True)
     username = Column(String(255), unique=True, nullable=False)
     hashed_password = Column(String(255), nullable=False)
     email = Column(String(255), unique=True, nullable=True)
     role_id = Column(Integer, ForeignKey("roles.role_id"), nullable=True)  # Associate user with a role
     disabled = Column(Boolean, default=True, nullable=True)
     created_at = Column(DateTime(timezone=True), server_default=func.now())
     updated_at = Column(TIMESTAMP, default="now()")
     # last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

     role = relationship("Role", back_populates="users")
     # Removed back_populates viewonly=True. in case, to make it less coupled. or you can totally remove it. 
     # predictions = relationship("Prediction", back_populates="user", viewonly=True)  
     # predictions = relationship("Prediction", back_populates="user")
     # error_logs = relationship("ErrorLog", back_populates="user")
     api_request_logs = relationship("APIRequestLog", back_populates="user")

class Role(Base):
     __tablename__ = "roles"

     role_id = Column(Integer, primary_key=True, index=True)
     role_name = Column(String(50), unique=True, nullable=False)
     role_description = Column(String(255), nullable=True)

     users = relationship("User", back_populates="role")

# CREATE TABLE roles (
#     role_id SERIAL PRIMARY KEY,
#     role_name VARCHAR(50),
#     role_description VARCHAR(255)
# );


# INSERT INTO roles (role_name, role_description) VALUES ('admin', 'Do the administrative activities');
# INSERT INTO roles (role_name, role_description) VALUES ('ordinary', 'Do the less sensitive activities such as prediction');

class APIRequestLog(Base):
     __tablename__ = "api_request_logs"

     request_id = Column(Integer, primary_key=True, index=True)
     endpoint = Column(String(255), nullable=False)
     request_method = Column(String(10), nullable=False)
     request_body = Column(Text, nullable=True)
     response_status = Column(Integer, nullable=False)
     response_time_ms = Column(Float, nullable=True)
     user_id = Column(Integer, ForeignKey("users.user_id"))
     ip_address = Column(String(45), nullable=True)
     timestamp = Column(DateTime(timezone=True), server_default=func.now())

     user = relationship("User", back_populates="api_request_logs")


class ErrorLog(Base):
     __tablename__ = "error_logs"

     error_id = Column(Integer, primary_key=True, index=True)
     error_type = Column(String(255), nullable=True)
     # The type of error (e.g., "Database Error," "Model Inference Error").
     error_message = Column(Text, nullable=False)
     # the_model_id = Column(Integer, ForeignKey("model_metadata.the_model_id"))
     the_model_id = Column(Integer, nullable=True)
     user_id = Column(Integer, nullable=True)
     # user_id = Column(Integer, ForeignKey("users.user_id"))
     timestamp = Column(DateTime(timezone=True), server_default=func.now())

     # model = relationship("ModelMetadata", back_populates="error_logs")
     # user = relationship("User", back_populates="error_logs")

