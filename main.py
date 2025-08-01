# main.py (Redis Version)
import os
import json
import time
import uuid
import redis # Import the redis library
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# --- 1. SETUP & CONSTANTS ---
load_dotenv()
EXPECTED_AUTH_TOKEN = "c88d7e70b6c77cd88271a48126bcd54761315985a275d864cd7e2b7ba342f1cf"

# --- NEW: Connect to Redis ---
redis_conn = redis.from_url(os.getenv("REDIS_URL"))

# --- 2. FASTAPI APP DEFINITION ---
app = FastAPI(title="Intelligent Query-Retrieval System")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=AnswerResponse)
async def process_queries(request: QueryRequest, http_request: Request):
    # --- Start the timer ---
    start_time = time.time()

    # --- A. Authentication ---
    auth_header = http_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    if token != EXPECTED_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

    # --- B. Job Creation ---
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id, # Include the job_id in the job data
        "document_url": request.documents,
        "questions": request.questions
    }

    # --- REPLACED: Push the job to the Redis queue ---
    redis_conn.lpush('job_queue', json.dumps(job_data))
    print(f"Job {job_id} created for document: {request.documents}")

    # --- C. Wait for Result ---
    print(f"Waiting for worker to process job {job_id}...")
    result_key = f"result:{job_id}"

    # --- REPLACED: Wait for the result from Redis ---
    # blpop is a "blocking" command that waits efficiently for the result.
    # We set a timeout of 10 minutes (600 seconds) for the web request.
    # On a free tier, PDF processing and indexing might take time.
    _, result_json = redis_conn.blpop(result_key, timeout=600)

    # --- Stop the timer and print the duration ---
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- Request for job {job_id} completed in {duration:.2f} seconds ---")

    if result_json:
        result_data = json.loads(result_json)
        return result_data
    else:
        # If blpop returns None, it means it timed out.
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Processing timed out after 10 minutes.")