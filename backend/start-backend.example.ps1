# ══════════════════════════════════════════════════════════════════════════════
#  FutureverseAI — Backend Startup (EXAMPLE)
#  Copy this to start-backend.ps1 and replace YOUR_* placeholders.
# ══════════════════════════════════════════════════════════════════════════════

# Start Weaviate (local Docker) — comment out if using Weaviate Cloud
Set-Location YOUR_PROJECT_ROOT\FutureverseAI\weaviate
docker-compose up -d
Start-Sleep -Seconds 5

Set-Location YOUR_PROJECT_ROOT\FutureverseAI\backend

# ── Weaviate (local) ──────────────────────────────────────────────────────────
$env:VECTOR_DB            = "weaviate"
$env:WEAVIATE_HTTP_HOST   = "localhost"
$env:WEAVIATE_GRPC_HOST   = "localhost"
$env:WEAVIATE_HTTP_PORT   = "8085"
$env:WEAVIATE_GRPC_PORT   = "50051"
$env:WEAVIATE_HTTP_SECURE = "false"
$env:WEAVIATE_GRPC_SECURE = "false"
$env:WEAVIATE_API_KEY     = ""

# ── Neo4j ─────────────────────────────────────────────────────────────────────
$env:NEO4J_URI      = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "YOUR_NEO4J_PASSWORD"
$env:NEO4J_DATABASE = "neo4j"

Write-Host "Starting FutureverseAI backend..." -ForegroundColor Cyan
py -3.11 -m uvicorn futureverse_ai.main:app --host 0.0.0.0 --port 8080
