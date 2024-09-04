import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres_user:postgres_password@db:5432/postgres_db")

    # Testing flag
    TESTING: bool = os.getenv("TESTING", "false").lower() == "false"

settings = Settings()