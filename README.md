# FutureverseAI Platform

An AI-powered adaptive learning platform combining RAG pipelines,
knowledge graphs, and behavioural assessment for UPSC preparation.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | SvelteKit + TypeScript + Tailwind CSS |
| Backend | FastAPI + SQLAlchemy + Alembic |
| Database | MySQL (production) / SQLite (dev) |
| Vector Store | Qdrant / Weaviate / pgvector (configurable) |
| Knowledge Graph | Neo4j + atlas-rag (AutoSchemaKG) |
| LLM | Ollama (local) + OpenAI-compatible APIs |
| Deployment | Docker Compose |

## Quick Start

### Prerequisites
- Docker Desktop
- Node.js 18+
- Python 3.11+
- MySQL 8.0+ (or use Docker)

### 1. Environment setup
```bash
cp .env.example .env
# Edit .env — fill in all required values (see Environment Variables below)
```

### 2. Start with Docker
```bash
docker-compose up -d
```

### 3. Backend (without Docker)
```bash
cd backend
pip install -r requirements.txt
alembic upgrade head
python -m futureverse_ai.main
```

### 4. Frontend (development)
```bash
npm install
npm run dev
# Dashboard at http://localhost:5173
# API at http://localhost:8080
```

### 5. Build frontend for production
```bash
npm run build
# Outputs to backend/futureverse_ai/static/
```

## Environment Variables

Copy `.env.example` to `.env`. Key variables:

| Variable | Description |
|---|---|
| `DATABASE_URL` | MySQL connection string |
| `WEBUI_SECRET_KEY` | JWT signing secret — change before deploying |
| `WEAVIATE_HTTP_HOST` | Weaviate host |
| `WEAVIATE_API_KEY` | Weaviate Cloud API key |
| `NEO4J_URI` | Neo4j bolt URI |
| `NEO4J_PASSWORD` | Neo4j password |
| `OPENAI_API_KEY` | OpenAI-compatible API key |
| `KG_EXTRACT_MODEL` | Ollama model for KG extraction |

Full variable list in `.env.example`.

## Project Structure

| Directory | Purpose |
|---|---|
| `src/` | SvelteKit frontend |
| `backend/futureverse_ai/` | FastAPI application |
| `backend/futureverse_ai/routers/` | API endpoint handlers |
| `backend/futureverse_ai/models/` | SQLAlchemy ORM models |
| `backend/futureverse_ai/migrations/` | Alembic migrations (34 total) |
| `backend/futureverse_ai/retrieval/` | RAG + knowledge graph pipeline |
| `cypress/` | End-to-end tests |

## Database Migrations

Every schema change requires an Alembic migration:
```bash
cd backend
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Related Projects

This platform is fed by the
[Agent Orchestration](https://github.com/futureverseai-in/agent-orchestration)
pipeline which processes NCERT PDFs into Q-Matrix knowledge graphs
and MCQs stored in MongoDB, then indexed into this platform.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch conventions,
commit format, and PR process.
