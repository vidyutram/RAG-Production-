**Day 7 Notes — Production RAG API**

**Project structure**
Split responsibilities across single purpose files — config, models, utils, ingestion, retrieval, generation, main. Each file does one thing. Changing the embedding model touches only utils. Changing the LLM touches only generation. This is why single responsibility matters in production.

**config.py**
`BaseSettings` from `pydantic_settings` reads values from environment variables and `.env` files automatically. Same Pydantic syntax you know — fields with types and defaults. Required fields like `openai_api_key` have no default so the app crashes at startup with a clear error if missing. Better than crashing mysteriously at runtime.

**models.py**
Request models are defensive — `extra="forbid"` rejects unexpected fields from untrusted external callers. Response models are descriptive — you construct them yourself so no need to guard against unexpected fields. Every boundary between outside world and your code has a typed Pydantic model.

**utils.py**
Shared logic that multiple modules need lives here. `get_embedding` lives in utils because both ingestion and retrieval need it. If it lived in one of those files, the other would have a weird dependency. Changing the embedding call happens in one place.

**ingestion.py**
Three responsibilities in order — chunk the text, embed in batches, upsert to Qdrant. `chunk_text` is pure Python, no async. `ensure_collection` guards collection creation so running the pipeline twice doesn't crash. Batching to 100 prevents OpenAI rate limit errors. `uuid.uuid4()` for point IDs ensures uniqueness across all documents. Payload stores text, source, and metadata so retrieval has everything the LLM needs.

**retrieval.py**
Embeds the question, builds an optional filter, searches Qdrant, converts raw `ScoredPoint` objects into typed `ChunkResult` objects. `with_payload=True` must be explicit or Qdrant returns scores and IDs with no text. `score_threshold` filters irrelevant chunks before they reach the LLM. Parameter names in the function call must match exactly what the function signature defines.

**generation.py**
`build_context` converts `list[ChunkResult]` into a readable string because the LLM receives text not Python objects. `generate_answer` awaits one complete response and returns it. `generate_answer_streaming` adds `stream=True` to the OpenAI call, loops with `async for`, and yields each token as it arrives. `if delta` guards against empty chunks in the stream. `temperature=0.2` keeps answers close to the source material.

**main.py**
Front door only — receives requests, calls the right functions, returns responses. No business logic. `/ingest` uses background tasks so the user doesn't wait for chunking and embedding. `/query` raises `HTTPException(404)` if no chunks pass the score threshold — better to tell the user clearly than pass empty context to the LLM. `/query/stream` passes the async generator directly into `StreamingResponse` — no `await`, no intermediate variable.

**How to derive function parameters**
Read what the function does. Find every piece of information it uses that doesn't come from inside itself. Those become parameters. Everything else comes from settings, other functions, or is created inside the function.

**How to derive what goes in a data structure**
Ask what the receiver needs. Qdrant needs an ID to manage the point, a vector for search, and a payload for retrieval. The LLM needs text, source for attribution, and metadata for context. Everything in a data structure exists because something downstream needs it.