How it works 

The recommendation flow happens in three steps:

1. Semantic Retrieval

All customer references are turned into embeddings using Ollama.
The user query is embedded the same way, and we use cosine similarity to find the closest matches.

This gives us a shortlist of candidates that are roughly relevant.

2. Re-Ranking (the smart part)

The shortlisted candidates are re-ranked using a CrossEncoder model
(BAAI/bge-reranker-base).

This step compares the query and each candidate together, which makes it very good at catching intent and keywords like:

“self-service”

“B2B”

“headless”

“scalability”

3. Pitch Generation

Once we have the best match, we pass:

the user query

the customer name

the original challenge

the project description

to an LLM, which generates a short, focused 2-sentence pitch explaining why this reference fits.

Tech stack

FastAPI – API framework

Ollama – embeddings + text generation

Sentence Transformers – CrossEncoder reranker

PyTorch

NumPy

scikit-learn

Simple, boring, reliable tools.

Project structure
.
├── main.py               # FastAPI app
├── references.json       # Customer case studies
├── README.md
└── requirements.txt

Customer references format

References are stored in references.json and look like this:

[
  {
    "customer": "Customer Name",
    "challenge": "What problem the customer had",
    "description": "What Intershop delivered and how"
  }
]


Nothing fancy — just structured text.

Getting started
Prerequisites

Python 3.9+

Ollama installed and running locally

Internet connection (first run downloads the reranker model)

Install dependencies
pip install -r requirements.txt

Pull Ollama models
ollama pull qwen3-embedding:0.6b
ollama pull llama3.1

Run the API
uvicorn main:app --reload


The API will be available at:

http://localhost:8000

API endpoints
POST /recommend

Returns the best customer reference plus a generated pitch.

Request

{
  "query": "We are looking for a self-service B2B commerce solution"
}


Response

{
  "query": "We are looking for a self-service B2B commerce solution",
  "top_recommendation": "ACME Corp",
  "pitch": "ACME Corp faced a similar challenge by enabling self-service ordering for complex B2B customers. Intershop helped them reduce sales friction while scaling their digital commerce platform.",
  "results": [
    {
      "customer": "ACME Corp",
      "rerank_score": 8.42,
      "cosine_score": 0.81
    }
  ]
}

GET /health

Simple health check to confirm the API is running.

{
  "status": "online",
  "models_loaded": true
}

Configuration

You can change the models at the top of main.py:

OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "llama3.1"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

Notes & tips

The reranker model (~90MB) is downloaded only once on startup

Embeddings are currently recomputed per request (fine for a PoC)

Debug output shows what the reranker actually sees
