"""CEO Chatbot FastAPI application.

The chatbot is stateless: no conversation history is stored server-side.
Each request must include the full conversation history in the request body.
This is required for Cloud Run multi-instance deployments, where requests
may be handled by different instances that share no memory.
"""
from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes import chat, health

app = FastAPI(
    title="CEO Chatbot",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
