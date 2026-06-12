import os
import httpx
from fastapi import FastAPI, Request
from telegram import Update
import io
import PyPDF2
import docx
from telegram.ext import Application, CommandHandler, MessageHandler, filters

RAG_API_URL = os.environ["RAG_API_URL"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

app = FastAPI()
pending_ingest = {}

ptb = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

async def start(update: Update, context):
    await update.message.reply_text(
        "Welcome to the RAG bot.\n\n"
        "Commands:\n"
        "/ingest <source_name> — then send your document text as the next message\n"
        "/query <your question> — get an answer from ingested documents"
        "/query <question> | <source> — search a specific document"
    )

async def ingest_command(update: Update, context):
    if not context.args:
        await update.message.reply_text("Usage: /ingest <source_name>")
        return
    source = " ".join(context.args)
    pending_ingest[update.effective_user.id] = source
    await update.message.reply_text(f"Got it. Now send the document text you want to ingest under source '{source}'.")

async def query_command(update: Update, context):
    if not context.args:
        await update.message.reply_text("Usage: /query <question> or /query <question> | <source>")
        return
    
    full_input = " ".join(context.args)
    
    if "|" in full_input:
        parts = full_input.split("|", 1)
        question = parts[0].strip()
        source_filter = parts[1].strip()
    else:
        question = full_input
        source_filter = None

    await update.message.reply_text("Searching...")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            payload = {"question": question, "top_k": 5}
            if source_filter:
                payload["source_filter"] = source_filter

            response = await client.post(
                f"{RAG_API_URL}/query",
                json=payload
            )
            if response.status_code == 404:
                await update.message.reply_text("No relevant documents found. Try ingesting something first.")
            else:
                data = response.json()
                await update.message.reply_text(data["answer"])
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")
async def handle_text(update: Update, context):
    user_id = update.effective_user.id
    if user_id in pending_ingest:
        source = pending_ingest.pop(user_id)
        text = update.message.text
        await update.message.reply_text("Ingesting your document...")
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                await client.post(
                    f"{RAG_API_URL}/ingest",
                    json={"text": text, "source": source}
                )
                await update.message.reply_text(f"Ingestion started for source '{source}'.")
            except Exception as e:
                await update.message.reply_text(f"Error: {str(e)}")
    else:
        await update.message.reply_text("Use /query <question> to ask something, or /ingest <source> to add a document.")

async def handle_document(update: Update, context):
    user_id = update.effective_user.id
    
    if user_id not in pending_ingest:
        await update.message.reply_text("Send /ingest <source_name> first before uploading a document.")
        return

    source = pending_ingest.pop(user_id)
    file = await update.message.document.get_file()
    file_name = update.message.document.file_name.lower()

    await update.message.reply_text("Downloading and extracting text...")

    file_bytes = await file.download_as_bytearray()
    buffer = io.BytesIO(bytes(file_bytes))

    try:
        if file_name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(buffer)
            text = "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        elif file_name.endswith(".txt"):
            text = buffer.read().decode("utf-8")
        elif file_name.endswith(".docx"):
            doc = docx.Document(buffer)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        else:
            await update.message.reply_text("File not supported. Please send a PDF, TXT, or DOCX file.")
            pending_ingest[user_id] = source
            return

        if not text.strip():
            await update.message.reply_text("Could not extract any text from the file. Try a different file.")
            return

        await update.message.reply_text(f"Extracted {len(text.split())} words. Ingesting... this may take 30-60 seconds.")

        async with httpx.AsyncClient(timeout=120) as client:
            await client.post(
                f"{RAG_API_URL}/ingest/sync",
                json={"text": text, "source": source}
            )
        await update.message.reply_text(f"Ingestion complete for source '{source}'. You can now query it.")

    except Exception as e:
        await update.message.reply_text(f"Error processing file: {str(e)}")

ptb.add_handler(CommandHandler("start", start))
ptb.add_handler(CommandHandler("ingest", ingest_command))
ptb.add_handler(CommandHandler("query", query_command))
ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
ptb.add_handler(MessageHandler(filters.Document.ALL, handle_document))

@app.on_event("startup")
async def startup():
    await ptb.initialize()
    await ptb.start()
    await ptb.bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")

@app.on_event("shutdown")
async def shutdown():
    await ptb.stop()
    await ptb.shutdown()

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, ptb.bot)
    await ptb.process_update(update)
    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok"}