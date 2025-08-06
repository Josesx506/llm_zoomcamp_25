import json

import requests
import uvicorn
from dependencies import SessionDep
from engines import create_db_and_tables
from fastapi import FastAPI, HTTPException, Request
from models import Conversations, Sample
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from utils import calculate_openai_cost, load_and_index_documents, rag
from fastapi.middleware.cors import CORSMiddleware

# Define rate limiter wrapper
limiter = Limiter(key_func=get_remote_address)

# Add the rate limiter to the app state
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware,allow_origins=["*"],
                   allow_credentials=True,allow_methods=["*"],allow_headers=["*"])


class PromptRequest(BaseModel):
    """
    Basemodel to handle json data types
    """
    prompt: str


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    load_and_index_documents("mini_wiki.csv")


@app.post("/generate")
@limiter.limit("10/minute")
async def generate(data:PromptRequest, request:Request, session: SessionDep):
    try:
        model = "gemma3:1b"# "gpt-4o-mini"# 
        resp, tokens, resp_time = rag(data.prompt,model)
        cost = calculate_openai_cost(model, tokens)
        new_sample_entry = Conversations(question=data.prompt,answer=resp,
                                         model_used=model,response_time=resp_time,
                                         openai_cost=cost, **tokens)
        new_sample_entry.insert(session)
        return {"response":resp, "id": new_sample_entry.id}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with llm: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)