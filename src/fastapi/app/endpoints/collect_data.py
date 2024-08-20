from fastapi import APIRouter, UploadFile, File, Depends, status, HTTPException
from typing import List, Dict
from app.auth.authentication import get_current_user
from app.utils.collect_data_utils import save_image, ClassLabels
from pathlib import Path

router = APIRouter()

# @router.post("/collect-data")
# async def collect_data(file: UploadFile = File(...), label: str = 'Apple__healthy', current_user: str = Depends(get_current_user)):
#      img = await file.read()
#      file_path = save_image(img, label, file.filename)
     
#      return {"message": f"Image saved successfully at {file_path}."}


@router.post("/collect-data")
async def collect_data(
     files: List[UploadFile] = File(...), 
     label: ClassLabels = ClassLabels.Apple_healthy, 
     current_user: str = Depends(get_current_user)):
     try:
          file_paths = []
          for file in files:
               img = await file.read()
               file_path = save_image(img, label.value, file.filename)
               file_paths.append(file_path)
          
          return {"message": [f"Image saved successfully at {file_path}" for file_path in file_paths]}
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to collect data successfully: {str(e)}")


@router.get("/list-images")
async def list_images() -> Dict[str, List[str]]:
     try:
          base_dir = Path("data")  # Base directory where images are stored
          image_dict = {}

          for label_dir in base_dir.iterdir():
               if label_dir.is_dir():
                    image_dict[label_dir.name] = [file.name for file in label_dir.iterdir() if file.is_file()]

          return image_dict
     except HTTPException as http_ex:
          raise http_ex
     except Exception as e:
          raise HTTPException(status_code=500, detail=f"Failed to retrieve images: {str(e)}")
