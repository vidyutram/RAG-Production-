import os
from langfuse import Langfuse
from langsmith import traceable

#LANGFUSE

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST")
)


async def traced_rag_request_langfuse(query: str) -> str:
    from app.api import retrieve_chunks, generate_answer

    trace = langfuse.trace(name="rag_request", input={"query": query})

    retrieve_span = trace.span(name="retrieve_chunks", input={"query": query})
    chunks = await retrieve_chunks(query)
    retrieve_span.end(output={"chunks": chunks})

    generate_span = trace.span(name="generate_answer", input={"query": query, "chunks": chunks})
    answer = await generate_answer(query, chunks)
    generate_span.end(output={"answer": answer})

    trace.update(output={"answer": answer})

    return answer


#LANGSMITH

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "rag-production")


@traceable(name="traced_rag_request")
async def traced_rag_request_langsmith(query: str) -> str:
    from app.api import retrieve_chunks, generate_answer 

    chunks = await retrieve_chunks(query)
    answer = await generate_answer(query, chunks)
    return answer