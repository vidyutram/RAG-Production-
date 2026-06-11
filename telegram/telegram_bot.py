import os
import json
import httpx
from fastapi import FastAPI, Request
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters

RAG_API_URL = os.environ["RAG_API_URL"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

bot = Bot(token=TELEGRAM_BOT_TOKEN)
app_telegram = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

app = FastAPI()

pending_ingest = {}

async def start(update: Update, context):
    await update.message.reply_text(
        "Welcome to the RAG bot.\n\n"
        "Commands:\n"
        "/ingest <source_name> — then send your document text as the next message\n"
        "/query <your question> — get an answer from ingested documents"
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
        await update.message.reply_text("Usage: /query <your question>")
        return
    question = " ".join(context.args)
    await update.message.reply_text("Searching...")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{RAG_API_URL}/query",
                json={"question": question, "top_k": 5}
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

app_telegram.add_handler(CommandHandler("start", start))
app_telegram.add_handler(CommandHandler("ingest", ingest_command))
app_telegram.add_handler(CommandHandler("query", query_command))
app_telegram.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

@app.on_event("startup")
async def startup():
    await app_telegram.initialize()
    await bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")

@app.on_event("shutdown")
async def shutdown():
    await app_telegram.shutdown()

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)
    await app_telegram.process_update(update)
    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok"}