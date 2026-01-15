# Docker Setup for Intershop Recommendation API

## Prerequisites
- Docker Desktop installed and running
- At least 8GB of free disk space (for Ollama models)

## Quick Start

### 1. Pull Ollama Models (do this first for faster startup)
```bash
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama ollama/ollama ollama pull qwen3-embedding:0.6b
docker run -d -v ollama:/root/.ollama ollama/ollama ollama pull llama3.1
```

### 2. Build and Start Services
```bash
docker-compose up --build
```

### 3. Access the Application
- FastAPI Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Ollama API: http://localhost:11434

## Services

### Ollama Service
- Downloads and runs large language models
- Provides embeddings and text generation
- Requires significant disk space for models

### Backend Service
- FastAPI application running on port 8000
- Depends on Ollama service being healthy
- Automatically restarts on failure

## Useful Commands

### View logs
```bash
docker-compose logs -f backend
docker-compose logs -f ollama
```

### Stop all services
```bash
docker-compose down
```

### Remove all data (including downloaded models)
```bash
docker-compose down -v
```

### Rebuild the backend image
```bash
docker-compose build --no-cache
```

### Test the API
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"query\":\"We need better e-commerce solutions\"}"
```

## Troubleshooting

### Models not loading
Ensure Ollama has downloaded the models first:
```bash
docker exec intershop_ollama ollama list
```

### Backend connection timeout
Wait for Ollama to be fully ready (check logs):
```bash
docker-compose logs ollama
```

### Out of memory
The services may need more than 8GB RAM. Check Docker Desktop settings.
