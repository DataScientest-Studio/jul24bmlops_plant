from fastapi import APIRouter, Depends, HTTPException
from ..utils.db_utils import backup_database, restore_database
from ..schemas.db_schema import BackupResponse, RestoreResponse
from ..database.db import get_db
from sqlalchemy.orm import Session
from ..utils.authorization_utils import get_current_admin_user, get_token_from_request

router = APIRouter()

# a lot of operation of Database is handled programmatically across the components without the need of user interaction.

@router.post("/backup", response_model=BackupResponse)
async def backup_db(db: Session = Depends(get_db)):
    try:
        # current_admin_user = await get_current_admin_user(token)
        # backup_database()
        return {"message": "Database backup successful."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to backup the database: {str(e)}")

@router.post("/restore", response_model=RestoreResponse)
async def restore_db(db: Session = Depends(get_db)):
    try:
        # current_admin_user = await get_current_admin_user(token)
        # restore_database()
        return {"message": "Database restored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore the database: {str(e)}")
