import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # General settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "f96979d3f80c071807f1309a00b3dc13e0d8558ab1f3cdca22650e8fd0e8a530")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 120))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres_user:postgres_password@db:5432/postgres_db")

    # Testing flag
    TESTING: bool = os.getenv("TESTING", "false").lower() == "false"

settings = Settings()