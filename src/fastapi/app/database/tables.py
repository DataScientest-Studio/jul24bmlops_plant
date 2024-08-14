from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text, TIMESTAMP, JSON
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
     __tablename__ = "users"
     
     user_id = Column(Integer, primary_key=True, index=True)
     username = Column(String(255), unique=True, nullable=False)
     hashed_password = Column(String(255), nullable=False)
     email = Column(String(255), unique=True, nullable=True)
     role_id = Column(Integer, ForeignKey("roles.role_id"), nullable=False)  # Associate user with a role
     disabled = Column(Boolean, default=True, nullable=True)
     created_at = Column(DateTime(timezone=True), server_default=func.now())
     updated_at = Column(TIMESTAMP, default="now()")
     # last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

     role = relationship("Role", back_populates="users")
     predictions = relationship("Prediction", back_populates="user")
     error_logs = relationship("ErrorLog", back_populates="user")
     api_request_logs = relationship("APIRequestLog", back_populates="user")


class Role(Base):
     __tablename__ = "roles"

     role_id = Column(Integer, primary_key=True, index=True)
     role_name = Column(String(50), unique=True, nullable=False)
     description = Column(String(255), nullable=True)

     users = relationship("User", back_populates="role")

class ModelMetadata(Base):
     __tablename__ = "model_metadata"
     
     model_id = Column(Integer, primary_key=True, index=True)
     model_name = Column(String)
     version = Column(String)
     training_data = Column(Text) # A description of the dataset used for training.
     training_start_time = Column(DateTime(timezone=True), nullable=True)
     training_end_time = Column(DateTime(timezone=True), nullable=True)
     accuracy = Column(Float, nullable=False)
     f1_score = Column(Float, nullable=True)
     precision = Column(Float, nullable=True)
     recall = Column(Float, nullable=True)
     training_loss = Column(Float, nullable=False)
     validation_loss = Column(Float, nullable=False)
     training_accuracy = Column(Float, nullable=True)
     validation_accuracy = Column(Float, nullable=True)
     training_params = Column(JSON, nullable=False)
     # Parameters used for training (e.g., learning rate, batch size).
     logs = Column(Text, nullable=True)
     created_at = Column(TIMESTAMP, default="now()")
     updated_at = Column(TIMESTAMP, default="now()")
     
     error_logs = relationship("ErrorLog", back_populates="model")
     predictions = relationship("Prediction", back_populates="model")
     ab_testing_results_a = relationship("ABTestingResult", foreign_keys=[ABTestingResult.model_a_id], back_populates="model_a")
     ab_testing_results_b = relationship("ABTestingResult", foreign_keys=[ABTestingResult.model_b_id], back_populates="model_b")
     ab_testing_winner = relationship("ABTestingResult", foreign_keys=[ABTestingResult.winning_model_id], back_populates="winning_model")


class ABTestingResult(Base):
     __tablename__ = "ab_testing_results"

     test_id = Column(Integer, primary_key=True, index=True)
     test_name = Column(String(255), nullable=False)
     model_a_id = Column(Integer, ForeignKey("model_metadata.model_id"))
     model_b_id = Column(Integer, ForeignKey("model_metadata.model_id"))
     metric_name = Column(String(255), nullable=False)
     # The metric being compared (e.g., accuracy, inference time).
     model_a_metric_value = Column(Float, nullable=False)
     model_b_metric_value = Column(Float, nullable=False)
     winning_model_id = Column(Integer, ForeignKey("model_metadata.model_id"))
     timestamp = Column(DateTime(timezone=True), server_default=func.now())

     model_a = relationship("ModelMetadata", foreign_keys=[model_a_id], back_populates="ab_testing_results_a")
     model_b = relationship("ModelMetadata", foreign_keys=[model_b_id], back_populates="ab_testing_results_b")
     winning_model = relationship("ModelMetadata", foreign_keys=[winning_model_id], back_populates="ab_testing_winner")


class Prediction(Base):
     __tablename__ = "predictions"
     
     prediction_id = Column(Integer, primary_key=True, index=True)
     user_id = Column(Integer, ForeignKey("users.user_id"))
     model_id = Column(Integer, ForeignKey("model_metadata.model_id"))
     image_path = Column(String, nullable=True)
     prediction = Column(JSON)
     top_5_prediction = Column(JSON, nullable=True)
     confidence = Column(Float)
     feedback_given = Column(Boolean, default=False)
     feedback_comment = Column(Text, nullable=True)
     predicted_at = Column(TIMESTAMP, default="now()")

     user = relationship("User", back_populates="predictions")
     model = relationship("ModelMetadata", back_populates="predictions")


class ErrorLog(Base):
     __tablename__ = "error_logs"

     error_id = Column(Integer, primary_key=True, index=True)
     error_type = Column(String(255), nullable=True)
     # The type of error (e.g., "Database Error," "Model Inference Error").
     error_message = Column(Text, nullable=False)
     model_id = Column(Integer, ForeignKey("model_metadata.model_id"))
     user_id = Column(Integer, ForeignKey("users.user_id"))
     timestamp = Column(DateTime(timezone=True), server_default=func.now())

     model = relationship("ModelMetadata", back_populates="error_logs")
     user = relationship("User", back_populates="error_logs")

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

    


