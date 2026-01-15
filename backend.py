import json
import torch
import ollama
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CONFIGURATION ---
# We keep Ollama for Embeddings + Generation
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b" 
LLM_MODEL = "llama3.1"

# CHANGED: Switched to a standard, reliable Reranker that works with CrossEncoder
# This model is small, fast, and excellent at catching keyword matches like "Self-service"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

# --- 2. GLOBAL STATE ---
state = {}

# --- 3. LIFESPAN DEFINITION ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[-] Loading Reranker: {RERANKER_MODEL_ID}...")
    try:
        # Load references
        with open('references.json', 'r') as f:
            state['references'] = json.load(f)
        
        # Load Model
        # This will download ~90MB from HuggingFace the first time
        state['reranker'] = CrossEncoder(RERANKER_MODEL_ID)
        
        print("[+] API and Models are ready.")
    except Exception as e:
        print(f"[!] Startup Error: {e}")
        state['references'] = []

    yield
    state.clear()

# --- 4. APP INITIALIZATION ---
app = FastAPI(title="Intershop Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. MODELS & HELPERS ---
class QueryRequest(BaseModel):
    query: str

def get_ollama_embedding(text):
    try:
        # Ensure we are embedding the same field structure as the text construction
        response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {e}")

def generate_pitch_logic(user_query, best_match):
    # FIXED: Added 'Challenge' to the prompt so the LLM knows WHY it matched
    prompt = f"""
    You are an Intershop Expert. A customer says: "{user_query}"
    
    The best case study match is: {best_match['customer']}
    The Challenge they solved: {best_match.get('challenge', 'N/A')}
    Project Details: {best_match['description']}
    
    Write a 2-sentence pitch explaining why this specific customer reference proves Intershop is the right choice for them.
    Focus on the specific problem (Challenge) they share.
    """
    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        return f"Pitch generation failed: {e}"

# --- 6. API ENDPOINTS ---
@app.post("/recommend")
async def recommend(request: QueryRequest):
    user_query = request.query
    references = state.get('references', [])
    reranker = state.get('reranker')

    if not references or reranker is None:
        raise HTTPException(status_code=500, detail="Models or References not loaded.")

    # --- STAGE 1: RETRIEVAL ---
    # We use all fields to ensure the embedding captures the full context
    corpus_texts = [f"{ref['customer']} {ref['challenge']} {ref['description']}" for ref in references]
    
    # Get Embeddings
    corpus_embeddings = [get_ollama_embedding(t) for t in corpus_texts]
    query_embedding = get_ollama_embedding(user_query)
    
    # Calculate Cosine Similarity
    sim_scores = cosine_similarity([query_embedding], corpus_embeddings)[0]
    
    # Get Top 30 Candidates
    top_indices = np.argsort(sim_scores)[::-1][:30]
    candidates = [references[i] for i in top_indices]
    candidate_cosine_scores = [sim_scores[i] for i in top_indices]

    # --- STAGE 2: RE-RANKING ---
    # FIXED: The Reranker MUST see the 'challenge' field to match the query "Self-service"
    # Format: [Query, Document_Text]
    rerank_pairs = []
    for c in candidates:
        # Construct the full text representation of the document
        doc_text = f"{c['customer']}. Challenge: {c['challenge']}. Description: {c['description']}"
        rerank_pairs.append([user_query, doc_text])

    # Debug print to verify what the model actually sees (Check your console!)
    print(f"DEBUG - Reranker Input Sample: {rerank_pairs[0]}")

    # Predict scores (higher is better)
    rerank_scores = reranker.predict(rerank_pairs, batch_size=1)

    # --- STAGE 3: FORMATTING ---
    final_results = []
    for i in range(len(candidates)):
        final_results.append({
            "data": candidates[i],
            "customer": candidates[i]['customer'],
            "rerank_score": max(float(rerank_scores[i]),0),
            "cosine_score": float(candidate_cosine_scores[i])
        })

    # Sort by Rerank Score
    final_results = sorted(final_results, key=lambda x: x['rerank_score'], reverse=True)
    
    # Generate Pitch for the Winner
    top_match = final_results[0]
    pitch = generate_pitch_logic(user_query, top_match['data'])

    return {
        "query": user_query,
        "top_recommendation": top_match['customer'],
        "pitch": pitch,
        "results": final_results[:5] # Return top 5
    }

@app.get("/health")
async def health():
    return {"status": "online", "models_loaded": "reranker" in state}
