from fastapi import APIRouter, Depends, status, HTTPException
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..auth.authentication import get_current_user
from ..utils.model import model

router = APIRouter()

@router.post("/retrain")
async def retrain(current_user: str = Depends(get_current_user)):
     # train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
     
     # train_generator = train_datagen.flow_from_directory(
     #      'data',
     #      target_size=(180, 180),
     #      batch_size=32,
     #      class_mode='sparse',
     #      subset='training'
     # )

     # validation_generator = train_datagen.flow_from_directory(
     #      'data',
     #      target_size=(180, 180),
     #      batch_size=32,
     #      class_mode='sparse',
     #      subset='validation'
     # )
    
     # model.fit(train_generator, validation_data=validation_generator, epochs=5)
     
     # model.save('models/plant_model.keras')
     
     return {"message": "Model retrained and saved."}
