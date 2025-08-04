# worker.py (FINAL-FINAL VERSION - REMOVED .tolist())
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import requests
import tempfile
import fitz
import pinecone
import redis
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import sys
from concurrent.futures import ThreadPoolExecutor

# --- 1. SETUP ---
def initialize_services():
    """Initializes all external services and returns client objects."""
    print("Worker starting up...")
    load_dotenv()
    
    redis_conn = redis.from_url(os.getenv("REDIS_URL"))
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    INDEX_NAME = "hackrx-index"
    EMBEDDING_DIMENSION = 1536
    
    print("Initializing Pinecone client...")
    pc = pinecone.Pinecone() 
    
    print(f"Checking for Pinecone index '{INDEX_NAME}'...")
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=INDEX_NAME, 
            dimension=EMBEDDING_DIMENSION, 
            metric='cosine',
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index created successfully. Please wait for initialization...")
        import time
        time.sleep(60)

    print(f"Connecting to index '{INDEX_NAME}'...")
    index = pc.Index(INDEX_NAME)

    print("Worker initialization complete. Connected to Pinecone.")
    return redis_conn, openai_client, openrouter_client, index, INDEX_NAME

# --- 2. HELPER FUNCTIONS ---

def get_embeddings(texts: list[str], openai_client) -> list[list[float]]:
    texts = [text.replace("\n", " ") for text in texts]
    response = openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [embedding.embedding for embedding in response.data]

def process_and_index_pdf(file_path: str, document_id: str, cancel_key: str, redis_conn, openai_client, index):
    print(f"Starting indexing: {document_id}")
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        if redis_conn.exists(cancel_key):
            raise InterruptedError(f"Job {document_id} canceled.")
            
        batch_chunks = chunks[i:i + batch_size]
        embeddings = get_embeddings(batch_chunks, openai_client)
        
        # FIX: Convert to float32 AND DO NOT convert back to a list
        embeddings_float32 = np.array(embeddings, dtype=np.float32)
        
        vectors_to_upsert = []
        # We need to convert the numpy array to a list for the zip function
        for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings_float32.tolist())):
            vectors_to_upsert.append({
                "id": f"{document_id}-chunk-{i+j}", 
                "values": embedding, 
                "metadata": {"text": chunk_text, "document_id": document_id}
            })
        # Upsert accepts the list of dictionaries with float32 values
        index.upsert(vectors=vectors_to_upsert)
    print(f"--- ✅ Finished indexing ---")

def find_most_similar_chunks_batch(query_embeddings: list[list[float]], document_id: str, index, top_k: int = 5) -> list[list[str]]:
    print(f"Batch querying Pinecone for {len(query_embeddings)} questions...")
    # FIX: Convert to float32 AND DO NOT convert back to a list
    queries_float32 = np.array(query_embeddings, dtype=np.float32)
    
    # The query function can handle the numpy array directly
    results = index.query(
        queries=queries_float32.tolist(), # Query needs a list of lists
        top_k=top_k, 
        include_metadata=True, 
        filter={"document_id": {"$eq": document_id}}
    )
    all_contexts = [[match['metadata']['text'] for match in res['matches']] for res in results['results']]
    print("--- ✅ Finished batch querying Pinecone ---")
    return all_contexts

def get_llm_answers_concurrently(questions: list[str], all_context_chunks: list[list[str]], openrouter_client) -> list[str]:
    print(f"Getting LLM answers for {len(questions)} questions concurrently...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        job_args = [(q, c, openrouter_client) for q, c in zip(questions, all_context_chunks)]
        results_iterator = executor.map(lambda p: get_llm_answer(*p), job_args)
        all_answers = list(results_iterator)
    print("--- ✅ Finished getting all LLM answers ---")
    return all_answers

def get_llm_answer(query: str, context_chunks: list[str], openrouter_client) -> str:
    if not context_chunks:
        return "Could not find relevant information in the document to answer this question."
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
    You are a precise assistant for answering questions about an insurance policy.
    Your answer MUST be based SOLELY on the context provided below.
    CRITICAL INSTRUCTION: When answering, you must first look for any specific exclusions, waiting periods, or limitations related to the user's question. If an exclusion clause is present, it OVERRIDES any general definition.
    FORMATTING INSTRUCTION: Provide your final answer as a single, clean paragraph. Do not use bullet points, markdown, or any special formatting.
    CONTEXT:
    {context}
    QUESTION: {query}
    ANSWER:
    """
    response = openrouter_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.2", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- 3. WORKER LOOP ---
def main_worker_loop(redis_conn, openai_client, openrouter_client, index, INDEX_NAME):
    PROCESSED_DOCS_SET_KEY = "processed_docs_v4"
    print(f"Worker started. Watching for jobs in Redis queue 'job_queue'...")
    while True:
        job_id, temp_pdf_path = None, None
        try:
            _, job_json = redis_conn.brpop('job_queue')
            job_data = json.loads(job_json)
            job_id = job_data["job_id"]
            cancel_key = f"cancel:{job_id}"
            if redis_conn.exists(cancel_key): continue
            print(f"Processing job: {job_id}")
            document_url = job_data["document_url"]
            document_id = os.path.basename(document_url.split('?')[0])
            questions = job_data["questions"]

            if not redis_conn.sismember(PROCESSED_DOCS_SET_KEY, document_id):
                response = requests.get(document_url)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(response.content)
                    temp_pdf_path = temp_pdf.name
                process_and_index_pdf(temp_pdf_path, document_id, cancel_key, redis_conn, openai_client, index)
                redis_conn.sadd(PROCESSED_DOCS_SET_KEY, document_id)
            
            question_embeddings = get_embeddings(questions, openai_client)
            if redis_conn.exists(cancel_key): raise InterruptedError(f"Job {job_id} canceled.")
            
            all_context_chunks = find_most_similar_chunks_batch(question_embeddings, document_id, index)
            if redis_conn.exists(cancel_key): raise InterruptedError(f"Job {job_id} canceled.")

            all_answers = get_llm_answers_concurrently(questions, all_context_chunks, openrouter_client)

            result_data = {"answers": all_answers}
            redis_conn.lpush(f"result:{job_id}", json.dumps(result_data))
            redis_conn.expire(f"result:{job_id}", 3600)
            print(f"Finished job: {job_id}")
        
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            if job_id:
                error_response = {"answers": [f"An error occurred: {str(e)}"]}
                redis_conn.lpush(f"result:{job_id}", json.dumps(error_response))
                redis_conn.expire(f"result:{job_id}", 3600)
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

if __name__ == "__main__":
    try:
        redis_conn, openai_client, openrouter_client, index, INDEX_NAME = initialize_services()
        main_worker_loop(redis_conn, openai_client, openrouter_client, index, INDEX_NAME)
    except Exception as e:
        print(f"FATAL: Worker failed to start. Error: {e}")
        sys.exit(1)
