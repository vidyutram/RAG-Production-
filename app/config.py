from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    openai_api_key : SecretStr
    qdrant_host : str = "qdrant" 
    qdrant_port : int = 6333
    collection_name : str = "documents"
    embedding_model : str = "text-embedding-3-small"
    embedding_dim : int = 1536
    chat_model : str = "gpt-4o-mini"
    chunk_size : int = 512
    chunk_overlap : int = 50
    retrieval_top_k : int = 5
    score_threshold : float = 0.7
    batch_size : int = 100
    model_config = {"env_file": ".env"}

settings = Settings()