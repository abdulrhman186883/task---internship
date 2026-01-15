import json
import torch
import ollama
import numpy as np
import requests
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"
RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
LLM_MODEL = "llama3.1"

# --- 1. DATA LOADING ---
try:
    REFERENCES = json.load(open('references.json'))
except FileNotFoundError:
    print("[!] Error: 'references.json' not found.")
    REFERENCES = []

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/")
        return response.status_code == 200
    except:
        return False

def get_ollama_embedding(text):
    try:
        response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[!] Embedding Error: {e}")
        return None

def generate_pitch(user_query, best_match):
    print(f"\n[-] Generating final pitch using Ollama ({LLM_MODEL})...")
    
    prompt = f"""
    You are an Intershop Expert. A customer says: "{user_query}"
    The best case study we found is: {best_match['customer']}.
    Details: {best_match['description']}
    
    Write a 2-sentence pitch explaining why this specific customer reference proves Intershop is the right choice for them.
    """
    
    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        print("\n🚀 FINAL RECOMMENDATION:")
        print("-" * 60)
        print(response['response'].strip())
        print("-" * 60)
    except Exception as e:
        print(f"[!] Pitch generation failed: {e}")

def run_local_recommendation(user_query):
    if not check_ollama_status():
        print("[!] ERROR: Ollama is not running. Please run 'ollama serve'.")
        return

    print(f"\n--- Processing Query: '{user_query}' ---")

    # 2. Stage 1: Retrieval (Cosine Similarity)
    print("[-] Stage 1: Computing Cosine Similarity...")
    corpus_texts = [f"{ref['customer']} {ref['challenge']} {ref['description']}" for ref in REFERENCES]
    
    corpus_embeddings = [get_ollama_embedding(t) for t in corpus_texts]
    query_embedding = get_ollama_embedding(user_query)
    
    if None in corpus_embeddings or query_embedding is None: return

    sim_scores = cosine_similarity([query_embedding], corpus_embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1][:20]
    
    candidates = [REFERENCES[i] for i in top_indices]
    candidate_cosine_scores = [sim_scores[i] for i in top_indices]

    # 3. Stage 2: Re-Ranking
    print(f"[-] Stage 2: Re-ranking with {RERANKER_MODEL_ID}...")
    try:
        reranker = CrossEncoder(RERANKER_MODEL_ID, trust_remote_code=True)
        
        if reranker.tokenizer.pad_token is None:
            reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
            reranker.model.config.pad_token_id = reranker.tokenizer.eos_token_id
        
        rerank_pairs = [[user_query, f"{c['customer']} {c['description']}"] for c in candidates]
        rerank_scores = reranker.predict(rerank_pairs, batch_size=1)

        final_results = []
        for i in range(len(candidates)):
            final_results.append({
                "data": candidates[i], # Keep original dict for the pitch
                "customer": candidates[i]['customer'],
                "final_score": float(rerank_scores[i]),
                "cosine_score": float(candidate_cosine_scores[i])
            })

        final_results = sorted(final_results, key=lambda x: x['final_score'], reverse=True)

        # 4. Display Table
        print("\n" + "="*60)
        print(f"{'CUSTOMER':<25} | {'RERANK':<10} | {'COSINE':<10}")
        print("-" * 60)
        for res in final_results[:5]: # Show top 5 in table
            print(f"{res['customer']:<25} | {res['final_score']:>10.4f} | {res['cosine_score']:>10.4f}")
        print("="*60)

        # 5. Generate the Pitch for the #1 result
        generate_pitch(user_query, final_results[0]['data'])

    except Exception as e:
        print(f"[!] Reranker Error: {e}")

if __name__ == "__main__":
    run_local_recommendation("customers worldwide after migrating to Intershop solution.")