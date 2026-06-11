import os
import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

RAG_API_URL = os.environ["RAG_API_URL"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to the RAG bot.\n\n"
        "Commands:\n"
        "/ingest <source_name> — then send your document text as the next message\n"
        "/query <your question> — get an answer from ingested documents"
    )

pending_ingest = {}

async def ingest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ingest <source_name>")
        return
    source = " ".join(context.args)
    pending_ingest[update.effective_user.id] = source
    await update.message.reply_text(f"Got it. Now send the document text you want to ingest under source '{source}'.")

async def query_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

if __name__ == "__main__":
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ingest", ingest_command))
    app.add_handler(CommandHandler("query", query_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)