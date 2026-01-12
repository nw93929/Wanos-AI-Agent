import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MONGO_URI: str = os.getenv("MONGO_URI")
    POSTGRES_URI: str = os.getenv("POSTGRES_URI")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    MAX_LOOPS: int = 3

settings = Settings()