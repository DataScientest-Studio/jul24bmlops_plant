from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text, TIMESTAMP, JSON, func, DateTime
from sqlalchemy.orm import relationship
from .db import Base


class Prediction(Base):
    __tablename__ = "predictions"
    
    prediction_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)  # No ForeignKey to users table in api_auth
    # we use http call to populate the user_id and the_model_id
    the_model_id = Column(Integer, nullable=True)
    image_path = Column(String, nullable=True)
    prediction = Column(JSON)
    top_5_prediction = Column(JSON, nullable=True)
    confidence = Column(Float)
    feedback_given = Column(Boolean, default=False)
    feedback_comment = Column(Text, nullable=True)
    predicted_at = Column(TIMESTAMP, default="now()")

    # since the there is tightly coupled relationship then we cannot use the 
    # user = relationship("User", back_populates="predictions")
    # model = relationship("ModelMetadata", back_populates="predictions")



     
     
     
