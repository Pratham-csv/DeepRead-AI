# main.py (Redis Version with Cancellation)
import os
import json
import time
import uuid
import redis
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

# --- 1. SETUP & CONSTANTS ---
load_dotenv()
EXPECTED_AUTH_TOKEN = "c88d7e70b6c77cd88271a48126bcd54761315985a275d864cd7e2b7ba342f1cf"
redis_conn = redis.from_url(os.getenv("REDIS_URL"))

# --- 2. FASTAPI APP DEFINITION ---
app = FastAPI(title="Intelligent Query-Retrieval System")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class JobSubmitResponse(BaseModel):
    job_id: str
    status: str

class AnswerResponse(BaseModel):
    answers: List[str]

# This endpoint is now split into two: one to submit the job, one to get the result.
# This is a better design for long-running tasks.
@app.post("/hackrx/run", response_model=JobSubmitResponse)
async def submit_job(request: QueryRequest, http_request: Request):
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
        "job_id": job_id,
        "document_url": request.documents,
        "questions": request.questions
    }
    
    # --- C. Push job to Redis queue ---
    redis_conn.lpush('job_queue', json.dumps(job_data))
    print(f"Job {job_id} created for document: {request.documents}")
    
    # --- D. Return the Job ID immediately ---
    # The client can now use this ID to check the status or get the result later.
    return {"job_id": job_id, "status": "queued"}

@app.get("/hackrx/result/{job_id}", response_model=AnswerResponse)
async def get_result(job_id: str):
    result_key = f"result:{job_id}"
    
    # Wait for a result using blocking pop, but with a shorter timeout for polling.
    # A more advanced version might just check the key without blocking.
    result_tuple = redis_conn.blpop(result_key, timeout=300) # Wait up to 5 minutes

    if result_tuple:
        _, result_json = result_tuple
        result_data = json.loads(result_json)
        return result_data
    else:
        # If no result is found, raise a 404 or a timeout error.
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Processing timed out or job not found.")

@app.post("/hackrx/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    Sets a cancellation flag in Redis for a specific job ID.
    """
    cancel_key = f"cancel:{job_id}"
    # Set the flag with a value of "1". Set an expiry (ex) of 10 minutes (600s)
    # so the flag doesn't stay in Redis forever.
    redis_conn.set(cancel_key, "1", ex=600)
    print(f"Cancellation requested for job {job_id}")
    return {"message": f"Cancellation signal sent for job {job_id}."}
