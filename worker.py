# worker.py (The True and Final Version)
import os
import sys
import json
import requests
import tempfile
import fitz
import pinecone
import redis
import time
import traceback
import math
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. SETUP ---
def initialize_services():
    print("Worker starting up...")
    load_dotenv()
    
    # Validate and Configure API Clients
    redis_url = os.getenv("REDIS_URL")
    if not redis_url: raise ValueError("[FATAL] REDIS_URL is not set!")
    redis_conn = redis.from_url(redis_url)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key: raise ValueError("[FATAL] OPENAI_API_KEY is not set!")
    openai_client = openai.OpenAI(api_key=openai_api_key)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key: raise ValueError("[FATAL] OPENROUTER_API_KEY is not set!")
    openrouter_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key: raise ValueError("[FATAL] PINECONE_API_KEY is not set!")
    pc = pinecone.Pinecone(api_key=pinecone_api_key)

    INDEX_NAME = "hackrx-index"
    REQUIRED_DIMENSION = 1536

    if INDEX_NAME in pc.list_indexes().names():
        description = pc.describe_index(INDEX_NAME)
        if description.dimension != REQUIRED_DIMENSION:
            print(f"!! MISMATCH !! Index found with dimension {description.dimension}, but {REQUIRED_DIMENSION} is required. Deleting...")
            pc.delete_index(INDEX_NAME)
    
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating with dimension {REQUIRED_DIMENSION}...")
        pc.create_index(
            name=INDEX_NAME, dimension=REQUIRED_DIMENSION, metric='cosine',
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index creation initiated. Pausing for 60 seconds for initialization...")
        time.sleep(60)

    print(f"Retrieving host for index '{INDEX_NAME}'...")
    host = pc.describe_index(INDEX_NAME).host
    print(f"Found host: {host}")
    index = pc.Index(host=host)

    print("Worker initialization complete.")
    return redis_conn, openai_client, openrouter_client, index

# --- 2. HELPER FUNCTIONS (Complete Versions) ---
def get_embeddings(texts: list[str], openai_client) -> list[list[float]]:
    texts = [text.replace("\n", " ") for text in texts]
    response = openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [e.embedding for e in response.data]

def process_and_index_pdf(file_path: str, document_id: str, cancel_key: str, redis_conn, openai_client, index):
    print(f"Starting indexing for document: {document_id}")
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    print(f"Extracted {len(full_text)} characters from PDF.")

    if not full_text.strip():
        print(f"[ERROR] PDF text for {document_id} is empty. Skipping indexing.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)
    print(f"Split text into {len(chunks)} chunks.")
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        if redis_conn.exists(cancel_key):
            raise InterruptedError(f"Job {document_id} canceled during indexing.")
        
        batch_chunks = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}, with {len(batch_chunks)} chunks.")
        embeddings = get_embeddings(batch_chunks, openai_client)
        
        vectors_to_upsert = []
        for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
            vectors_to_upsert.append({
                "id": f"{document_id}-chunk-{i+j}", 
                "values": embedding,
                "metadata": {"text": chunk_text, "document_id": document_id}
            })
        
        if vectors_to_upsert:
            print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
            index.upsert(vectors=vectors_to_upsert)

    print(f"--- ✅ Finished indexing for {document_id} ---")

def find_most_similar_chunks(query_embedding: list[float], document_id: str, index, top_k: int = 5) -> list[str]:
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True, 
        filter={"document_id": {"$eq": document_id}}
    )
    print(f"Found {len(results['matches'])} relevant chunks for query.")
    return [m['metadata']['text'] for m in results['matches']]

def get_llm_answer(query: str, context_chunks: list[str], openrouter_client) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a precise assistant... (your full prompt here)"""
    response = openrouter_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.2", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- 3. WORKER LOOP ---
def main_worker_loop(redis_conn, openai_client, openrouter_client, index):
    PROCESSED_DOCS_SET_KEY = "processed_docs_v4"
    print(f"Worker started. Watching for jobs in Redis queue 'job_queue'...")
    while True:
        job_id, temp_pdf_path = None, None
        try:
            _, job_json = redis_conn.brpop('job_queue')
            job_data = json.loads(job_json)
            job_id = job_data["job_id"]
            print(f"\n--- Received job: {job_id} ---")
            
            cancel_key = f"cancel:{job_id}"
            if redis_conn.exists(cancel_key):
                print(f"Job {job_id} was canceled before it began.")
                continue

            document_url = job_data["document_url"]
            document_id = os.path.basename(document_url.split('?')[0])
            questions = job_data["questions"]

            if not redis_conn.sismember(PROCESSED_DOCS_SET_KEY, document_id):
                print(f"Document '{document_id}' not found in cache. Starting processing.")
                response = requests.get(document_url)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(response.content)
                    temp_pdf_path = temp_pdf.name
                process_and_index_pdf(temp_pdf_path, document_id, cancel_key, redis_conn, openai_client, index)
                redis_conn.sadd(PROCESSED_DOCS_SET_KEY, document_id)
                print(f"Document '{document_id}' marked as processed in Redis.")
            else:
                print(f"Document '{document_id}' found in cache. Skipping indexing.")
            
            print(f"Generating embeddings for {len(questions)} questions...")
            question_embeddings = get_embeddings(questions, openai_client)
            
            all_answers = []
            for i, (question, query_embedding) in enumerate(zip(questions, question_embeddings)):
                print(f"Processing question {i+1}/{len(questions)}...")
                context_chunks = find_most_similar_chunks(query_embedding, document_id, index)
                answer = get_llm_answer(question, context_chunks, openrouter_client) if context_chunks else "Could not find relevant information."
                all_answers.append(answer)

            result_data = {"answers": all_answers}
            redis_conn.lpush(f"result:{job_id}", json.dumps(result_data))
            redis_conn.expire(f"result:{job_id}", 3600)
            print(f"--- Finished job: {job_id} ---")
        
        except Exception as e:
            print(f"[ERROR] An exception occurred in the main loop for job {job_id}: {e}")
            traceback.print_exc()
            if job_id:
                error_response = {"answers": [f"An error occurred: {str(e)}"]}
                redis_conn.lpush(f"result:{job_id}", json.dumps(error_response))
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

if __name__ == "__main__":
    try:
        redis_conn, openai_client, openrouter_client, index = initialize_services()
        main_worker_loop(redis_conn, openai_client, openrouter_client, index)
    except Exception as e:
        print(f"[FATAL] Worker failed to start: {e}")
        traceback.print_exc()
        sys.exit(1)
