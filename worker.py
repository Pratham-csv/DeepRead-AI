# worker.py (Corrected and Robust Version)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import requests
import tempfile
import fitz
import pinecone
import redis
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import sys # Import sys to exit gracefully on failure

# --- 1. SETUP ---
# In worker.py, replace the initialize_services function with this one.

def initialize_services():
    """Initializes all external services and returns client objects."""
    print("Worker starting up...")
    load_dotenv()

    # --- Configure API clients ---
    print("Connecting to Redis...")
    redis_conn = redis.from_url(os.getenv("REDIS_URL"))
    
    print("Configuring OpenAI client...")
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("Configuring OpenRouter client...")
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    print("Configuring Pinecone client...")
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # --- Connect to Pinecone Index ---
    INDEX_NAME = "hackrx-index"
    EMBEDDING_DIMENSION = 1536
    
    print(f"Checking for Pinecone index '{INDEX_NAME}'...")
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=INDEX_NAME, 
            dimension=EMBEDDING_DIMENSION, 
            metric='cosine',
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index created successfully. Please wait a moment for it to initialize...")
        import time
        time.sleep(60) # Give Pinecone a minute to initialize the new index

    # SOLUTION: Automatically get the host from the index description
    print(f"Getting host for index '{INDEX_NAME}'...")
    host = pc.describe_index(INDEX_NAME).host
    print(f"Found host: {host}")
    
    index = pc.Index(host=host)

    print("Worker initialization complete.")
    return redis_conn, openai_client, openrouter_client, index, INDEX_NAME

# --- 2. HELPER FUNCTIONS ---
# (Helper functions remain largely the same, but will now receive clients as arguments)

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
        
        vectors_to_upsert = []
        for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
            vectors_to_upsert.append({
                "id": f"{document_id}-chunk-{i+j}", 
                "values": embedding,
                "metadata": {"text": chunk_text, "document_id": document_id}
            })
        index.upsert(vectors=vectors_to_upsert)
    print(f"--- ✅ Finished indexing ---")

def find_most_similar_chunks(query_embedding: list[float], document_id: str, index, top_k: int = 5) -> list[str]:
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"document_id": {"$eq": document_id}})
    return [match['metadata']['text'] for match in results['matches']]

def get_llm_answer(query: str, context_chunks: list[str], openrouter_client) -> str:
    # SOLUTION: This now uses the client passed as an argument, not a new one.
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a precise assistant... (rest of your prompt)"""
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

            if redis_conn.exists(cancel_key):
                print(f"Job {job_id} was canceled before it began.")
                continue

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
            
            # SOLUTION: Get all question embeddings in one efficient batch call
            question_embeddings = get_embeddings(questions, openai_client)
            
            all_answers = []
            for question, query_embedding in zip(questions, question_embeddings):
                if redis_conn.exists(cancel_key):
                    raise InterruptedError(f"Job {job_id} canceled during processing.")
                
                context_chunks = find_most_similar_chunks(query_embedding, document_id, index)
                answer = get_llm_answer(question, context_chunks, openrouter_client) if context_chunks else "Could not find relevant information."
                all_answers.append(answer)

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
    # SOLUTION: Add a top-level try/except block.
    # This will catch ANY error during initialization and print it,
    # so you never have a silent crash again.
    try:
        redis_conn, openai_client, openrouter_client, index, INDEX_NAME = initialize_services()
        main_worker_loop(redis_conn, openai_client, openrouter_client, index, INDEX_NAME)
    except Exception as e:
        print(f"FATAL: Worker failed to start. Error: {e}")
        sys.exit(1) # Exit with an error code
