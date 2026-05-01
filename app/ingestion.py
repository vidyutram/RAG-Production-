from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings
import asyncio, uuid
from app.utils import get_embedding

qdrant_client = AsyncQdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)

def chunk_text(text: str, chunk_size:int, chunk_overlap:int):
    chunk = text.split()
    chunks = []
    for c in range(0, len(chunk), chunk_size - chunk_overlap):
        batch = " ".join(chunk[c : c + chunk_size])
        chunks.append(batch)
    return chunks

async def ensure_collection():
    exists = await qdrant_client.collection_exists(settings.collection_name)
    if not exists:
        await qdrant_client.create_collection(
            collection_name = settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim, 
                distance=Distance.COSINE
                ),
        )
        await qdrant_client.create_payload_index(
            collection_name = settings.collection_name,
            field_name = "source",
            field_schema = "keyword"
        )

async def ingest_document(text: str, source: str, metadata: dict| None = None):
    await ensure_collection()

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    for i in range(0, len(chunks), settings.batch_size):
        batch = chunks[i : i + settings.batch_size]

        embeddings = await asyncio.gather(
            *[get_embedding(chunk) for chunk in batch]
        )
        points = [
            PointStruct(
                id= str(uuid.uuid4()),
                vector= embeddings[j],
                payload = {
                    "text": batch[j],
                    "source": source,
                    "metadata": metadata or {} 
                }
            )
            for j in range(len(batch))
        ]
        await qdrant_client.upsert(
            collection_name = settings.collection_name,
            points = points
        )
