from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text, TIMESTAMP, JSON, func, DateTime
from sqlalchemy.orm import relationship
from .db import Base

class ABTestingResult(Base):
    __tablename__ = "ab_testing_results"

    test_id = Column(Integer, primary_key=True, index=True)
    test_name = Column(String(255), nullable=False)
    model_a_id = Column(Integer, ForeignKey("model_metadata.the_model_id"))
    model_b_id = Column(Integer, ForeignKey("model_metadata.the_model_id"))
    metric_name = Column(String(255), nullable=False)
    # The metric being compared (e.g., accuracy, inference time).
    model_a_metric_value = Column(Float, nullable=False)
    model_b_metric_value = Column(Float, nullable=False)
    winning_the_model_id = Column(Integer, ForeignKey("model_metadata.the_model_id"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    model_a = relationship("ModelMetadata", foreign_keys=[model_a_id], back_populates="ab_testing_results_a")
    model_b = relationship("ModelMetadata", foreign_keys=[model_b_id], back_populates="ab_testing_results_b")
    winning_model = relationship("ModelMetadata", foreign_keys=[winning_the_model_id], back_populates="ab_testing_winner")



class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    
    the_model_id = Column(Integer, primary_key=True, index=True)
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
    ab_testing_winner = relationship("ABTestingResult", foreign_keys=[ABTestingResult.winning_the_model_id], back_populates="winning_model")

