import json

import requests
import uvicorn
from auth import API_KEY
from dependencies import AuthKeyDep, SessionDep
from engines import create_db_and_tables
from fastapi import FastAPI, HTTPException
from models import Sample
from pydantic import BaseModel
from utils import load_and_index_documents, rag

app = FastAPI()

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
async def generate(request:PromptRequest, session: SessionDep, x_api_key:AuthKeyDep):
    try:
        API_KEY[x_api_key] -= 1 # Subtract a credit
        resp = rag(request.prompt)
        print(resp)
        new_sample_entry = Sample(answer=resp)
        new_sample_entry.insert(session)
        print(new_sample_entry.id)
        return resp
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with llm: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)