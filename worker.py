# worker.py (Redis Version)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import requests
import tempfile
import fitz
import pinecone
import redis # Import the redis library
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# --- 1. SETUP & MODEL LOADING ---
print("Worker starting up...")
load_dotenv()

# --- NEW: Connect to Redis ---
# This will connect to Redis using the URL from your environment variables.
# It works for both local testing and for the Redis instance on a deployment platform.
redis_conn = redis.from_url(os.getenv("REDIS_URL"))

# Configure the OpenRouter client
client = openai.OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Configure Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define constants
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_NAME = "hackrx-index"
GENERATION_MODEL = "mistralai/mistral-7b-instruct-v0.2"
EMBEDDING_DIMENSION = 384
PROCESSED_DOCS_SET_KEY = "processed_docs" # Key for our Redis set

# Load local models and connect to Pinecone
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

# --- 2. RAG HELPER FUNCTIONS ---

# --- REPLACED: Use Redis to check if a document is indexed ---
def is_document_indexed(document_id: str) -> bool:
    return redis_conn.sismember(PROCESSED_DOCS_SET_KEY, document_id)

# --- REPLACED: Use Redis to mark a document as indexed ---
def mark_document_as_indexed(document_id: str):
    redis_conn.sadd(PROCESSED_DOCS_SET_KEY, document_id)

def get_embedding(text):
    text = text.replace("\n", " ")
    return embedding_model.encode(text).tolist()

def process_and_index_pdf(file_path: str, document_id: str):
    print(f"Starting indexing: {document_id}")
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        embeddings = [get_embedding(text) for text in batch_chunks]
        vectors_to_upsert = []
        for j, chunk_text in enumerate(batch_chunks):
            vectors_to_upsert.append({
                "id": f"{document_id}-chunk-{i+j}", "values": embeddings[j],
                "metadata": {"text": chunk_text, "document_id": document_id}
            })
        index.upsert(vectors=vectors_to_upsert)
    print(f"--- ✅ Finished indexing ---")

def find_most_similar_chunks(query_embedding, document_id: str, top_k=5):
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"document_id": {"$eq": document_id}})
    return [match['metadata']['text'] for match in results['matches']]

def get_llm_answer(query, context_chunks):
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

# --- 3. WORKER LOOP ---
def main_worker_loop():
    print("Worker started. Watching for jobs in Redis queue 'job_queue'...")
    while True:
        try:
            # --- REPLACED: Wait for a job from the Redis queue ---
            # brpop is a "blocking" command. It waits efficiently until a job is available.
            _, job_json = redis_conn.brpop('job_queue')
            job_data = json.loads(job_json)

            job_id = job_data["job_id"]
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
                process_and_index_pdf(temp_pdf_path, document_id)
                os.unlink(temp_pdf_path)
                mark_document_as_indexed(document_id)

            all_answers = []
            for question in questions:
                query_embedding = get_embedding(question)
                context_chunks = find_most_similar_chunks(query_embedding, document_id)
                if not context_chunks:
                    answer = "Could not find relevant information in the document."
                else:
                    answer = get_llm_answer(question, context_chunks)
                all_answers.append(answer)

            # --- REPLACED: Push the result back to Redis ---
            result_data = {"answers": all_answers}
            result_key = f"result:{job_id}"
            redis_conn.lpush(result_key, json.dumps(result_data))
            # Set an expiration so the result doesn't stay in Redis forever
            redis_conn.expire(result_key, 3600)
            print(f"Finished job: {job_id}")

        except Exception as e:
            print(f"Error processing job: {e}")
            error_response = {"answers": [f"An error occurred: {e}"]}
            if 'job_id' in locals():
                result_key = f"result:{job_id}"
                redis_conn.lpush(result_key, json.dumps(error_response))
                redis_conn.expire(result_key, 3600)

if __name__ == "__main__":
    main_worker_loop()