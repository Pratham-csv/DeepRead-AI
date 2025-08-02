# worker.py (Redis Version with Cancellation)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import requests
import tempfile
import fitz
import pinecone
import redis
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# --- 1. SETUP & MODEL LOADING (Same as before) ---
print("Worker starting up...")
load_dotenv()
redis_conn = redis.from_url(os.getenv("REDIS_URL"))
client = openai.OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_NAME = "hackrx-index"
GENERATION_MODEL = "mistralai/mistral-7b-instruct-v0.2"
EMBEDDING_DIMENSION = 384
PROCESSED_DOCS_SET_KEY = "processed_docs"
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("Connecting to Pinecone index...")
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
    pc.create_index(
        name=INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric='cosine',
        spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)
print("Models loaded. Worker is ready.")

# --- 2. RAG HELPER FUNCTIONS (Same as before) ---
def is_document_indexed(document_id: str) -> bool:
    return redis_conn.sismember(PROCESSED_DOCS_SET_KEY, document_id)
def mark_document_as_indexed(document_id: str):
    redis_conn.sadd(PROCESSED_DOCS_SET_KEY, document_id)
def get_embedding(text):
    text = text.replace("\n", " ")
    return embedding_model.encode(text).tolist()
# In worker.py - The updated function
def process_and_index_pdf(file_path: str, document_id: str, cancel_key: str):
    print(f"Starting indexing: {document_id}")
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)
    
    batch_size = 100 
    for i in range(0, len(chunks), batch_size):
        if redis_conn.exists(cancel_key):
            print(f"Job canceled during indexing for {document_id}")
            raise InterruptedError(f"Job {document_id} canceled.")
            
        batch_chunks = chunks[i:i + batch_size]
        
        # --- OPTIMIZATION: Process the entire batch at once ---
        embeddings = embedding_model.encode(batch_chunks).tolist()
        
        vectors_to_upsert = []
        for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
            vectors_to_upsert.append({
                "id": f"{document_id}-chunk-{i+j}", 
                "values": embedding,
                "metadata": {"text": chunk_text, "document_id": document_id}
            })
        index.upsert(vectors=vectors_to_upsert)
    print(f"--- ✅ Finished indexing ---")
def find_most_similar_chunks(query_embedding, document_id: str, top_k=5):
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"document_id": {"$eq": document_id}})
    return [match['metadata']['text'] for match in results['matches']]
def get_llm_answer(query, context_chunks):
    # (This function is the same as before)
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
    response = client.chat.completions.create(
        model=GENERATION_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().replace("\n", " ")

# --- 3. WORKER LOOP (Updated with Cancellation Logic) ---
def main_worker_loop():
    print("Worker started. Watching for jobs in Redis queue 'job_queue'...")
    while True:
        job_id = None
        temp_pdf_path = None
        try:
            _, job_json = redis_conn.brpop('job_queue')
            job_data = json.loads(job_json)
            job_id = job_data["job_id"]
            cancel_key = f"cancel:{job_id}"

            # --- CHECKPOINT 1: Before starting ---
            if redis_conn.exists(cancel_key):
                print(f"Job {job_id} was canceled before it began.")
                continue # Skip to the next job

            print(f"Processing job: {job_id}")
            document_url = job_data["document_url"]
            document_id = os.path.basename(document_url.split('?')[0])
            questions = job_data["questions"]

            if not is_document_indexed(document_id):
                response = requests.get(document_url)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(response.content)
                    temp_pdf_path = temp_pdf.name
                
                # --- CHECKPOINT 2: After download, before indexing ---
                if redis_conn.exists(cancel_key):
                    print(f"Job {job_id} was canceled after download.")
                    raise InterruptedError(f"Job {job_id} canceled.")

                process_and_index_pdf(temp_pdf_path, document_id, cancel_key)
                mark_document_as_indexed(document_id)
            
            all_answers = []
            for question in questions:
                 # --- CHECKPOINT 3: Before each question ---
                if redis_conn.exists(cancel_key):
                    print(f"Job {job_id} was canceled during question processing.")
                    raise InterruptedError(f"Job {job_id} canceled.")
                
                query_embedding = get_embedding(question)
                context_chunks = find_most_similar_chunks(query_embedding, document_id)
                answer = get_llm_answer(question, context_chunks) if context_chunks else "Could not find relevant information in the document."
                all_answers.append(answer)

            result_data = {"answers": all_answers}
            redis_conn.lpush(f"result:{job_id}", json.dumps(result_data))
            redis_conn.expire(f"result:{job_id}", 3600)
            print(f"Finished job: {job_id}")
        
        except InterruptedError as e:
            print(e) # Just print the cancellation message and move on
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            if job_id:
                error_response = {"answers": [f"An error occurred: {e}"]}
                redis_conn.lpush(f"result:{job_id}", json.dumps(error_response))
                redis_conn.expire(f"result:{job_id}", 3600)
        finally:
            # This block ensures the temp file is always deleted, even if the job is canceled or fails
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

if __name__ == "__main__":
    main_worker_loop()
