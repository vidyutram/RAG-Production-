from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.models import IngestRequest, QueryRequest, QueryResponse
from app.ingestion import ingest_document
from app.retrieval import retrieve_chunks
from app.generation import generate_answer, generate_answer_streaming
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production RAG API")

@app.get("/health")
async def health():
    return{"status": "ok"}

@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_document, request.text, request.source, request.metadata)
    return {"status": "ingestion started", "source":request.source}

@app.post("/query")
async def query(request: QueryRequest):
    results = await retrieve_chunks(
        question = request.question, 
        top_k = request.top_k, 
        source_filter = request.source_filter
    )
    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    
    answer = await generate_answer(request.question, results)
    return QueryResponse(question=request.question, chunks_used=results, answer=answer)

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    results = await retrieve_chunks(
        question = request.question, 
        top_k = request.top_k, 
        source_filter = request.source_filter
    )
    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    
    return StreamingResponse(
        generate_answer_streaming(request.question, results),
        media_type="text/event-stream"
        )