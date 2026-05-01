from openai import AsyncOpenAI
from app.config import settings
from app.models import ChunkResult

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = "with the chunks of text provided, generate in a human readable format.answer only from the provided context and say clearly if the context doesn't contain enough information."

def build_context(chunks: list[ChunkResult]):
    return "\n\n".join(
        f"[Source:{chunk.source} | Score: {chunk.score:.3f}\n{chunk.text}]"
        for chunk in chunks    
    )

async def generate_answer(question:str, chunks: list[ChunkResult]):
    context = build_context(chunks)
    response = await openai_client.chat.completions.create(
        model = settings.chat_model,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature= 0.2,
    )
    return response.choices[0].message.content

async def generate_answer_streaming(question:str, chunks: list[ChunkResult]):
    context = build_context(chunks)
    stream = await openai_client.chat.completions.create(
        model = settings.chat_model,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature= 0.2,
        stream = True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
            