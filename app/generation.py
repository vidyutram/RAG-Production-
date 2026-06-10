from groq import AsyncGroq
from app.config import settings
from app.models import ChunkResult

groq_client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())

SYSTEM_PROMPT = "With the chunks of text provided, generate in a human readable format. Answer only from the provided context and say clearly if the context doesn't contain enough information."

def build_context(chunks: list[ChunkResult]):
    return "\n\n".join(
        f"[Source: {chunk.source} | Score: {chunk.score:.3f}\n{chunk.text}]"
        for chunk in chunks
    )

async def generate_answer(question: str, chunks: list[ChunkResult]) -> str:
    context = build_context(chunks)
    response = await groq_client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

async def generate_answer_streaming(question: str, chunks: list[ChunkResult]):
    context = build_context(chunks)
    stream = await groq_client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta