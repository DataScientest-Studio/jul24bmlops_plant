import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "f96979d3f80c071807f1309a00b3dc13e0d8558ab1f3cdca22650e8fd0e8a530")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

settings = Settings()

