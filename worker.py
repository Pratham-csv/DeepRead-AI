import os
import sys
import json
import requests
import tempfile
import fitz
import pinecone
import redis
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import math
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. SETUP ---

def initialize_services():
    print("Worker starting up...")
    load_dotenv()

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("[FATAL] REDIS_URL is not set!")
        sys.exit(1)
    redis_conn = redis.from_url(redis_url)
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[FATAL] OPENAI_API_KEY is not set!")
        sys.exit(1)
    openai_client = openai.OpenAI(api_key=openai_api_key)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("[FATAL] OPENROUTER_API_KEY is not set!")
        sys.exit(1)
    openrouter_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("[FATAL] PINECONE_API_KEY is not set!")
        sys.exit(1)
    pc = pinecone.Pinecone(api_key=pinecone_api_key)

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
        print("Index created. Waiting 60s to initialize...")
        import time
        time.sleep(60)

    print(f"Retrieving Pinecone host for '{INDEX_NAME}'...")
    host = pc.describe_index(INDEX_NAME).host
    print(f"Found host: {host}")
    index = pc.Index(host=host)

    print("Worker initialization complete.")
    return redis_conn, openai_client, openrouter_client, index, INDEX_NAME

# --- 2. HELPER FUNCTIONS ---

def is_valid_embedding(embedding, dim=1536):
    if not isinstance(embedding, list) or len(embedding) != dim:
        return False
    for v in embedding:
        if not isinstance(v, float) or not math.isfinite(v):
            return False
    return True

def get_embeddings(texts: list[str], openai_client, dim=1536) -> list[list[float]]:
    texts = [text.replace("\n", " ") for text in texts]
    response = openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = []
    for idx, embedding_obj in enumerate(response.data):
        emb = embedding_obj.embedding
        emb = np.array(emb, dtype=np.float32).tolist()
        if not is_valid_embedding(emb, dim):
            print(f"[ERROR] Embedding {idx} invalid (len={len(emb) if isinstance(emb, list) else 'N/A'}); skipped.")
            continue
        if idx == 0:
            print(f"[DEBUG] Sample embedding (first 5 values): {emb[:5]}")
        embeddings.append(emb)
    print(f"[DEBUG] Generated {len(embeddings)}/{len(texts)} embeddings.")
    return embeddings

def process_and_index_pdf(file_path: str, document_id: str, cancel_key: str, redis_conn, openai_client, index):
    print(f"Starting indexing: {document_id}")
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    print(f"[DEBUG] Extracted text length: {len(full_text)}")
    print(f"[DEBUG] First 300 chars: {repr(full_text[:300])}")

    if not full_text.strip():
        print("[ERROR] Extracted PDF text is empty. Document may be scanned/image-based or encrypted.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)
    print(f"[DEBUG] Number of text chunks: {len(chunks)}")
    if chunks:
        print(f"[DEBUG] First chunk: {repr(chunks[0])}")

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        if redis_conn.exists(cancel_key):
            raise InterruptedError(f"Job {document_id} canceled.")
        batch_chunks = chunks[i:i + batch_size]
        embeddings = get_embeddings(batch_chunks, openai_client)
        if not embeddings:
            print("[ERROR] No embeddings were created for this batch!")
            continue

        vectors_to_upsert = []
        for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
            vectors_to_upsert.append({
                "id": f"{document_id}-chunk-{i+j}",
                "values": embedding,
                "metadata": {"text": chunk_text, "document_id": document_id}
            })
        if vectors_to_upsert:
            print(f"[DEBUG] Upserting {len(vectors_to_upsert)} vectors with document_id: {document_id}")
            for v in vectors_to_upsert[:3]:  # print sample up to 3 vectors
                print(f"  Vector ID: {v['id']}, document_id metadata: {v['metadata']['document_id']}")
            index.upsert(vectors=vectors_to_upsert)
    print(f"--- ✅ Finished indexing ---")

def find_most_similar_chunks(query_embedding: list[float], document_id: str, index, dim=1536, top_k: int = 5) -> list[str]:
    if not is_valid_embedding(query_embedding, dim):
        print(f"[ERROR] Query embedding invalid (len={len(query_embedding) if isinstance(query_embedding, list) else 'N/A'}); skipping similarity search.")
        return []
    print(f"[DEBUG] Querying Pinecone for document_id={document_id}, embedding length={len(query_embedding)}")

    # Try querying WITH filter
    results_with_filter = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"document_id": {"$eq": document_id}}
    )
    print(f"[DEBUG] Pinecone matches returned with filter: {len(results_with_filter['matches'])}")

    if len(results_with_filter['matches']) > 0:
        return [match['metadata']['text'] for match in results_with_filter['matches']]

    # If no matches, try AGAIN without filter for diagnostics
    print("[DEBUG] No matches with filter; querying without filter for diagnostics...")
    results_without_filter = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    print(f"[DEBUG] Pinecone matches returned without filter: {len(results_without_filter['matches'])}")

    return [match['metadata']['text'] for match in results_without_filter['matches']]

def get_llm_answer(query: str, context_chunks: list[str], openrouter_client) -> str:
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

def main_worker_loop(redis_conn, openai_client, openrouter_client, index, INDEX_NAME, dim=1536):
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
            
            question_embeddings = get_embeddings(questions, openai_client, dim)
            if len(question_embeddings) != len(questions):
                print(f"[WARNING] Some embeddings could not be created for the given questions. Proceeding with valid ones.")

            all_answers = []
            for i, question in enumerate(questions):
                if i >= len(question_embeddings):
                    all_answers.append("Could not generate embedding for this question.")
                    continue
                query_embedding = question_embeddings[i]
                if redis_conn.exists(cancel_key):
                    raise InterruptedError(f"Job {job_id} canceled during processing.")
                context_chunks = find_most_similar_chunks(query_embedding, document_id, index, dim)
                if not context_chunks:
                    print(f"[DEBUG] No relevant contexts found for question {i}.")
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
    try:
        redis_conn, openai_client, openrouter_client, index, INDEX_NAME = initialize_services()
        main_worker_loop(redis_conn, openai_client, openrouter_client, index, INDEX_NAME)
    except Exception as e:
        print(f"FATAL: Worker failed to start. Error: {e}")
        sys.exit(1)
