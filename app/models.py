from pydantic import BaseModel, Field
from typing import Optional

class IngestRequest(BaseModel):
    text: str
    source: str
    metadata: Optional[dict] = None
    model_config = {"extra": "forbid"}

class QueryRequest(BaseModel):
    question : str 
    source_filter: Optional[str]= None
    top_k : int = Field(le =20, ge = 1, default= 5)
    model_config = {"extra": "forbid"}

class ChunkResult(BaseModel):
    text: str
    source: str
    score: float
    metadata: Optional[dict] = None
    model_config = {"extra": "forbid"}

class QueryResponse(BaseModel):
    question : str
    chunks_used : list[ChunkResult]
    answer: str