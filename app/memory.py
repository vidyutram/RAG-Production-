import uuid
from datetime import datetime
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings
from app.utils import get_embedding
import asyncio

MEMORY_COLLECTION = "user_memory"
MEMORY_DIM = 768

qdrant_client = AsyncQdrantClient(
    url=f"https://{settings.qdrant_host}",
    api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
)

async def ensure_memory_collection():
    exists = await qdrant_client.collection_exists(MEMORY_COLLECTION)
    if not exists:
        await qdrant_client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=VectorParams(size=MEMORY_DIM, distance=Distance.COSINE),
        )
        await qdrant_client.create_payload_index(
            collection_name=MEMORY_COLLECTION,
            field_name="user_id",
            field_schema="keyword"
        )
        await qdrant_client.create_payload_index(
            collection_name=MEMORY_COLLECTION,
            field_name="topic",
            field_schema="keyword"
        )

async def store_memory(user_id: str, question: str, answer: str, topic: str):
    await ensure_memory_collection()
    text = f"Q: {question}\nA: {answer}"
    embedding = await get_embedding(text)

    # deduplication — check if near identical memory exists
    existing = await qdrant_client.query_points(
        collection_name=MEMORY_COLLECTION,
        query=embedding,
        query_filter=Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="topic", match=MatchValue(value=topic))
        ]),
        limit=1,
        score_threshold=0.97,
    )
    if existing.points:
        return  # near duplicate exists, skip storing

    await qdrant_client.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "question": question,
                    "answer": answer,
                    "topic": topic,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        ]
    )

async def detect_topic(question: str) -> str:
    topics = ["magic", "characters", "plot", "places", "general"]
    query_embedding = await get_embedding(question)
    
    import numpy as np
    query_vec = np.array(query_embedding)
    
    best_topic = "general"
    best_score = -1

    for topic in topics:
        topic_embedding = await get_embedding(topic)
        topic_vec = np.array(topic_embedding)
        score = float(np.dot(query_vec, topic_vec) /
                     (np.linalg.norm(query_vec) * np.linalg.norm(topic_vec)))
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic

async def retrieve_memories(user_id: str, question: str, top_k: int = 3) -> list[dict]:
    await ensure_memory_collection()
    query_embedding = await get_embedding(question)

    results = await qdrant_client.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        query_filter=Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]),
        limit=top_k,
        score_threshold=0.5,
    )

    return [
        {
            "question": r.payload["question"],
            "answer": r.payload["answer"],
            "topic": r.payload["topic"]
        }
        for r in results.points
    ]