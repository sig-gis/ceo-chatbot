"""CEO Chatbot FastAPI application.

The chatbot is stateless: no conversation history is stored server-side.
Each request must include the full conversation history in the request body.
This is required for Cloud Run multi-instance deployments, where requests
may be handled by different instances that share no memory.
"""
import os
import uvicorn
from fastapi import FastAPI

from ceo_rag_chatbot.lifespan import lifespan
from ceo_rag_chatbot.routes import chat, health

app = FastAPI(
    title="CEO Chatbot",
    version="0.1.0",
    lifespan=lifespan,
)

def run() -> None:
    uvicorn.run(
        "ceo_rag_chatbot.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        log_level="info",
    )

if __name__ == "__main__":
    run()

app.include_router(health.router)
app.include_router(chat.router)
