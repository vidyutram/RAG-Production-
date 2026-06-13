from groq import AsyncGroq
from app.config import settings
from app.models import ChunkResult

groq_client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())

SYSTEM_PROMPT = "With the chunks of text provided, generate in a human readable format. Answer only from the provided context and say clearly if the context doesn't contain enough information."

def build_context(chunks: list[ChunkResult]) -> str:
    return "\n\n".join(
        f"[Source: {chunk.source} | Score: {chunk.score:.3f}\n{chunk.text}]"
        for chunk in chunks
    )

def build_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""
    lines = ["Relevant past conversation context:"]
    for m in memories:
        lines.append(f"Q: {m['question']}\nA: {m['answer']}")
    return "\n\n".join(lines)

async def generate_answer(question: str, chunks: list[ChunkResult], memories: list[dict] = None) -> str:
    context = build_context(chunks)
    memory_context = build_memory_context(memories or [])

    user_content = f"Context:\n{context}"
    if memory_context:
        user_content += f"\n\n{memory_context}"
    user_content += f"\n\nQuestion: {question}"

    response = await groq_client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

async def generate_answer_streaming(question: str, chunks: list[ChunkResult], memories: list[dict] = None):
    context = build_context(chunks)
    memory_context = build_memory_context(memories or [])

    user_content = f"Context:\n{context}"
    if memory_context:
        user_content += f"\n\n{memory_context}"
    user_content += f"\n\nQuestion: {question}"

    stream = await groq_client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2,
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

def build_short_term_context(turns: list[dict]) -> str:
    if not turns:
        return ""
    lines = ["Recent conversation history:"]
    for turn in turns:
        lines.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
    return "\n\n".join(lines)

async def generate_answer(question: str, chunks: list[ChunkResult], memories: list[dict] = None, short_term_turns: list[dict] = None) -> str:
    context = build_context(chunks)
    memory_context = build_memory_context(memories or [])
    short_term_context = build_short_term_context(short_term_turns or [])

    user_content = f"Context:\n{context}"
    if short_term_context:
        user_content += f"\n\n{short_term_context}"
    if memory_context:
        user_content += f"\n\n{memory_context}"
    user_content += f"\n\nQuestion: {question}"

    response = await groq_client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

async def generate_answer_streaming(question: str, chunks: list[ChunkResult], memories: list[dict] = None, short_term_turns: list[dict] = None):
    context = build_context(chunks)
    memory_context = build_memory_context(memories or [])
    short_term_context = build_short_term_context(short_term_turns or [])

    user_content = f"Context:\n{context}"
    if short_term_context:
        user_content += f"\n\n{short_term_context}"
    if memory_context:
        user_content += f"\n\n{memory_context}"
    user_content += f"\n\nQuestion: {question}"

    stream = await groq_client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2,
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta