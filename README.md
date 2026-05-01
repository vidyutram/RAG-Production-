# Production RAG API

A fully containerized Retrieval Augmented Generation API built with FastAPI, Qdrant, and OpenAI.

## Stack
- FastAPI — async REST API
- Qdrant — vector database for semantic search
- OpenAI — embeddings and generation
- Docker Compose — containerized deployment
- Pydantic v2 — request validation and typed contracts

## Features
- Document ingestion with batched embedding and chunking
- Semantic search with configurable score threshold
- Metadata filtering on vector search
- Streaming and non-streaming query endpoints
- Background task ingestion so clients don't wait
- Payload indexing for fast filtered search

## Running locally
cp .env.example .env  # add your OpenAI API key
docker-compose up --build

## Endpoints
POST /ingest — chunk and index a document
POST /query — retrieve and generate answer
POST /query/stream — streaming answer generation
GET /health — health check
