# worker.py (BATCH-OPTIMIZED VERSION)
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
import sys
from concurrent.futures import ThreadPoolExecutor # <-- ADDED THIS IMPORT

# --- 1. SETUP ---
# This section is unchanged.
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
        time.sleep(60)

    print(f"Getting host for index '{INDEX_NAME}'...")
    host = pc.describe_index(INDEX_NAME).host
    print(f"Found host: {host}")
    
    index = pc.Index(host=host)

    print("Worker initialization complete.")
    return redis_conn, openai_client, openrouter_client, index, INDEX_NAME

# --- 2. HELPER FUNCTIONS ---
# The original helper functions are unchanged.
# We add NEW batch-specific functions.

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

# --- START: NEW BATCH HELPER FUNCTIONS ---

def find_most_similar_chunks_batch(query_embeddings: list[list[float]], document_id: str, index, top_k: int = 5) -> list[list[str]]:
    """
    Performs a batch query to Pinecone to find context for multiple questions at once.
    """
    print(f"Batch querying Pinecone for {len(query_embeddings)} questions...")
    # The pinecone client can take a list of vectors directly
    results = index.query(
        queries=query_embeddings, 
        top_k=top_k, 
        include_metadata=True, 
        filter={"document_id": {"$eq": document_id}}
    )
    
    # Process the batch results into a list of context lists
    all_contexts = []
    for res in results['results']:
        context_chunks = [match['metadata']['text'] for match in res['matches']]
        all_contexts.append(context_chunks)
    print("--- ✅ Finished batch querying Pinecone ---")
    return all_contexts

def get_llm_answers_concurrently(questions: list[str], all_context_chunks: list[list[str]], openrouter_client) -> list[str]:
    """
    Calls the LLM for all questions concurrently using a thread pool.
    """
    print(f"Getting LLM answers for {len(questions)} questions concurrently...")
    
    all_answers = []
    # We use a ThreadPoolExecutor to make all API calls at the same time
    with ThreadPoolExecutor(max_workers=10) as executor:
        # We prepare the arguments for each call to the original get_llm_answer function
        job_args = []
        for question, context_chunks in zip(questions, all_context_chunks):
            job_args.append((question, context_chunks, openrouter_client))

        # The executor.map function runs the calls in parallel
        # We define a small lambda function to unpack the arguments for each thread
        results_iterator = executor.map(lambda p: get_llm_answer(*p), job_args)
        
        # Collect results
        all_answers = list(results_iterator)

    print("--- ✅ Finished getting all LLM answers ---")
    return all_answers

# --- END: NEW BATCH HELPER FUNCTIONS ---

# This is the original single-item function, now used by the concurrent helper
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


# --- 3. WORKER LOOP (MODIFIED FOR BATCHING) ---
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
            
            # --- START: BATCH-OPTIMIZED LOGIC ---
            # The FOR-LOOP is now gone. We process everything in batches.

            # 1. Get all question embeddings in one API call (no change, was already optimal)
            question_embeddings = get_embeddings(questions, openai_client)
            
            # Check for cancellation before the next expensive step
            if redis_conn.exists(cancel_key):
                raise InterruptedError(f"Job {job_id} canceled after embedding.")
            
            # 2. Get all context chunks from Pinecone in one API call
            all_context_chunks = find_most_similar_chunks_batch(question_embeddings, document_id, index)
            
            # Check for cancellation before the final expensive step
            if redis_conn.exists(cancel_key):
                raise InterruptedError(f"Job {job_id} canceled after context retrieval.")

            # 3. Get all answers from the LLM concurrently
            all_answers = get_llm_answers_concurrently(questions, all_context_chunks, openrouter_client)

            # --- END: BATCH-OPTIMIZED LOGIC ---

            result_data = {"answers": all_answers}
            redis_conn.lpush(f"result:{job_id}", json.dumps(result_data))
            redis_conn.expire(f"result:{job_id}", 3600)
            print(f"Finished job: {job_id}")
        
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            if job_id:
                # Ensure the error message is a list to match the expected AnswerResponse model
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
