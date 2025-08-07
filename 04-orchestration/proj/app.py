import json
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd

import requests
import uvicorn
from dependencies import SessionDep
from engines import create_db_and_tables
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from models import Conversations, Messages
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from utils import (calculate_openai_cost, get_recent_messages,
                   load_and_index_documents, rag)

# Define rate limiter wrapper
limiter = Limiter(key_func=get_remote_address)


class PromptRequest(BaseModel):
    """
    Basemodel to handle json data types
    """
    prompt: str
    conversation_id: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    create_db_and_tables()
    load_and_index_documents("mini_wiki.csv")
    yield  # The application will run here
    # Shutdown events
    print("Application shutting down...")


# Add the rate limiter to the app state
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware,allow_origins=["*"],
                   allow_credentials=True,allow_methods=["*"],allow_headers=["*"])


@app.get("/conversations")
async def get_conversations(session: SessionDep):
    """
    Get all existing conversations.
        - Pagination can be implemented for faster load times
    """
    try:
        conversations = (
            session.query(Conversations)
            .with_entities(Conversations.id, Conversations.title)
            .order_by(Conversations.timestamp.asc())
            .all()
        )
        df = pd.DataFrame(conversations, columns=["conv_id", "title"])
        return {"conversations": df.to_dict(orient="records")}
    except requests.RequestException as e:
        raise HTTPException(status_code=404, detail=f"{str(e)}")


@app.get("/conversations/{conv_id}/messages")
async def get_conversation_messages(conv_id: int, session: SessionDep):
    """
    Get all prior conversations associated with a specific conversation
    """
    try:
        messages = (
            session.query(Messages)
            .with_entities(Messages.id, Messages.question, Messages.answer, Messages.like_dislike)
            .filter(Messages.conv_id == conv_id)
            .order_by(Messages.timestamp.asc())
            .all()
        )
        df = pd.DataFrame(messages, columns=["msg_id", "query", "response", "like_dislike"])
        if df.empty:
            raise HTTPException(status_code=404, detail="Conversation messages not found")
        return {"messages": df.to_dict(orient="records")}
    except requests.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Error retrieving conversation: {str(e)}")


@app.post("/generate")
@limiter.limit("10/minute")
async def generate(data: PromptRequest, request: Request, session: SessionDep):
    """
    Save an active conversationn message
    """
    try:
        model = "gemma3:1b"

        # Handle conversation (new or existing)
        if data.conversation_id:
            # Get conversation or raise 404
            conv = session.get(Conversations, data.conversation_id)
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            # Create new conversation
            conv = Conversations(title=data.prompt[:50])
            conv.insert(session)

        # Fetch last 3 messages from this conversation
        history = get_recent_messages(session, conv.id)

        # Run RAG + history
        resp, tokens, resp_time = rag(data.prompt, history, model)
        cost = calculate_openai_cost(model, tokens)

        # Save message
        new_msg = Messages(
            conv_id=conv.id,
            question=data.prompt,
            answer=resp,
            model_used=model,
            response_time=resp_time,
            openai_cost=cost,
            **tokens
        )
        new_msg.insert(session)

        return {
            "response": resp,
            "conversation_id": conv.id,
            "msg_id": new_msg.id,
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM: {str(e)}")


# async def generate(data:PromptRequest, request:Request, session: SessionDep):
#     try:
#         model = "gemma3:1b"# "gpt-4o-mini"# 
#         resp, tokens, resp_time = rag(data.prompt,model)
#         cost = calculate_openai_cost(model, tokens)
#         new_sample_entry = Conversations(question=data.prompt,answer=resp,
#                                          model_used=model,response_time=resp_time,
#                                          openai_cost=cost, **tokens)
#         new_sample_entry.insert(session)
#         return {"response":resp, "id": new_sample_entry.id}
#     except requests.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Error communicating with llm: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)