from app.config import settings
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key = settings.openai_api_key)

async def get_embedding(text : str):
    response = await client.embeddings.create(
    model = settings.embedding_model,
    input = text,
    )
    return response.data[0].embedding