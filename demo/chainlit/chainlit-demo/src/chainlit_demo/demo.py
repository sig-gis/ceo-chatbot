import asyncio
import chainlit as cl
import httpx
import json

SERVER_URL = "http://localhost:8080"
TIMEOUT = 15

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="What is CEO?",
            message="What is CEO?",
            # icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Reviewing plots",
            message="How can I review plots that have been already interpreted?",
            # icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Quality Control feature",
            message="Can you explain how to use the Quality Control feature under Plot Design?",
            # icon="/public/idea.svg",
        )
    ]

@cl.on_chat_start
async def main():
    """
    Glorified health check for now.
    Repeatedly queries the server for 3 minutes every 15 seconds.
    """
    client = httpx.AsyncClient()
    cl.user_session.set("client", client)
    status = 0
    num_retries = 0
    while status != 200 and num_retries < 12:
        res = await client.get(f"{SERVER_URL}/healthz")
        status = res.status_code
        if status != 200:
            print("Client not ready, waiting for 15 seconds before trying again!")
            await asyncio.sleep(15)
            num_retries += 1
            
    if not status == 200:
        await cl.Message(content="Server did not respond").send()
        await client.aclose()
        raise RuntimeError("Server is not active after 3 minutes of waiting")
    
    cl.user_session.set(
        "message_history",
        [],
    )
    print("Server started")

def _update_history(user_message, assistant_message):
    cl.user_session.get("message_history").append(
        {
            "role": "user",
            "content": user_message
        }
    )
    cl.user_session.get("message_history").append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def message(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    # Call the tool
    response = await send_message(message.content)
    response_content = response.json()
    _update_history(message.content, response_content["answer"])
    await cl.Message(content=response_content["answer"]).send()

@cl.step(type="tool")
async def send_message(msg: dict):
    client = cl.user_session.get("client")
    message_history = cl.user_session.get("message_history")
    llm_payload = {
        "query": msg,
        "history": message_history if message_history else []
    }
    response = await client.post(f"{SERVER_URL}/chat", 
                                    json=llm_payload,
                                    timeout=TIMEOUT)
    return response

@cl.on_chat_end
async def end():
    await cl.user_session.get("client").aclose()
    print("goodbye", cl.user_session.get("id"))
    