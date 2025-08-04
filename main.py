# main.py (Corrected Secure Version)
import os
import json
import uuid
import redis
import asyncio
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# --- 1. SETUP & CONSTANTS ---
load_dotenv()
# SOLUTION: Load the token securely from environment variables.
# This prevents leaking your secret key in your source code.
EXPECTED_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
if not EXPECTED_AUTH_TOKEN:
    raise ValueError("API_AUTH_TOKEN environment variable not set.")

redis_conn = redis.from_url(os.getenv("REDIS_URL"))

# --- 2. FASTAPI APP DEFINITION ---
app = FastAPI(title="Intelligent Query-Retrieval System")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=AnswerResponse)
async def process_queries_and_wait(request: QueryRequest, http_request: Request):
    # --- A. Authentication ---
    auth_header = http_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    if token != EXPECTED_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

    # (The rest of the file is unchanged and correct)
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "document_url": request.documents,
        "questions": request.questions
    }
    
    redis_conn.lpush('job_queue', json.dumps(job_data))
    print(f"Job {job_id} created for document: {request.documents}")

    result_key = f"result:{job_id}"
    cancel_key = f"cancel:{job_id}"
    
    try:
        while True:
            if await http_request.is_disconnected():
                print(f"Client disconnected for job {job_id}. Sending cancellation signal.")
                redis_conn.set(cancel_key, "1", ex=600)
                raise HTTPException(status_code=499, detail="Client closed request.")

            result_json = redis_conn.rpop(result_key)
            if result_json:
                result_data = json.loads(result_json)
                print(f"Result found for job {job_id}. Returning to client.")
                return result_data

            await asyncio.sleep(1)

    except asyncio.CancelledError:
        print(f"Request loop cancelled for job {job_id}. Sending cancellation signal.")
        redis_conn.set(cancel_key, "1", ex=600)
        raise
