from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    groq_api_key: SecretStr
    jina_api_key: SecretStr
    qdrant_host: str = "qdrant"
    qdrant_api_key: SecretStr | None = None
    collection_name: str = "documents"
    embedding_model: str = "jina-embeddings-v2-base-en"
    embedding_dim: int = 768
    chat_model: str = "llama-3.1-8b-instant"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    score_threshold: float = 0.7
    batch_size: int = 100
    model_config = {"env_file": ".env"}

settings = Settings()