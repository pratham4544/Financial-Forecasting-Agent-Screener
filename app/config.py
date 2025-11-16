from pydantic import BaseSettings

class Settings(BaseSettings):
    MYSQL_DSN: str = "mysql+pymysql://user:password@localhost:3306/tcs_forecast"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"

settings = Settings()
