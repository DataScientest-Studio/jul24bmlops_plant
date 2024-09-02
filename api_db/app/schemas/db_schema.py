from pydantic import BaseModel

class BackupResponse(BaseModel):
    message: str

class RestoreResponse(BaseModel):
    message: str

    class Config:
        from_attributes = True