import httpx
from app.config import settings

async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.jina_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.embedding_model,
                "input": [text],
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]