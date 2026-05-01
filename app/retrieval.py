from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.utils import get_embedding
from app.config import settings
from app.models import ChunkResult

qdrant_client = AsyncQdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)

async def retrieve_chunk(question:str, top_k:int, source_filter:str | None = None):
    query_vector = await get_embedding(question)

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key= "source",
                    match= MatchValue(value = source_filter)
                )
            ]
        )
        results = await qdrant_client.search(
            collection_name = settings.collection_name,
            query_vector= query_vector,
            limit = top_k,
            query_filter = source_filter,
            with_payload = True,
            score_thresehold = settings.score_threshold
        )

        return[
            ChunkResult(
                text = r.payload["text"],
                source =r.payload["source"],
                score = r.score,
                metadata=r.payload.get("metadata")
            )
            for r in results
        ]