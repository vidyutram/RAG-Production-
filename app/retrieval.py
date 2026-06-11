from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.utils import get_embedding
from app.config import settings
from app.models import ChunkResult

qdrant_client = AsyncQdrantClient(
    url=f"https://{settings.qdrant_host}",
    api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
)

async def retrieve_chunks(question: str, top_k: int, source_filter: str | None = None):
    query_vector = await get_embedding(question)

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter)
                )
            ]
        )

    results = await qdrant_client.query_points(
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
        score_threshold=settings.score_threshold
    )

    return [
        ChunkResult(
            text=r.payload["text"],
            source=r.payload["source"],
            score=r.score,
            metadata=r.payload.get("metadata")
        )
        for r in results.points
    ]