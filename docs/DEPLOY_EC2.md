# FutureverseAI — AWS EC2 Deployment Guide

A complete, step-by-step runbook to deploy the FutureverseAI RAG platform on a single GPU EC2 instance using Docker Compose. For app architecture see [`RAG_PIPELINE.md`](RAG_PIPELINE.md); for general run/config see [`../SETUP.md`](../SETUP.md).

> **Key difference from your local box:** EC2 GPUs (A10G / L4 / L40S = NVIDIA Ampere/Ada) run **Ollama on CUDA directly** — the Vulkan / SM_120 workaround in `start_backend.ps1` is **local-only** and is NOT used here.

---

## 0. Architecture

Everything runs in containers on one GPU instance, wired by `docker-compose.prod.yaml`:

```
                Internet (443)
                     |
              [ Caddy / Nginx ]  TLS, reverse proxy
                     |
              [ backend :8080 ] FastAPI + SvelteKit (GPU: embeddings + reranker)
                /    |    |   \____________________
               /     |    |        \               \
        [ Ollama ] [Weaviate] [ Neo4j ]   [ MySQL ]  [ Redis ]
         (GPU LLMs)  (vectors)  (KG)        (app DB)  (cache)
```

- **GPU is shared** by Ollama (LLMs) + the PyTorch embedder/reranker.
- All data lives in **Docker named volumes** on EBS.
- Only **80/443** are public; everything else is internal to the instance.

---

## 1. Sizing — instance & storage

**VRAM budget** (single GPU): `qwen2.5:32b` ≈ 20 GB (Q4) + `bge-large-en-v1.5` embedder ≈ 1.5 GB + `bge-reranker-v2-m3` ≈ 2.5 GB + overhead ⇒ **~24–26 GB peak**.

| Instance | GPU | VRAM | Use it for | Rough $/hr (verify) |
|---|---|---|---|---|
| **g6e.xlarge** | 1× L40S | **48 GB** | **Recommended** — 32B answer model + embed + rerank with headroom | ~$1.8–2.5 |
| g5.2xlarge | 1× A10G | 24 GB | Budget — only with a smaller answer model (`qwen2.5:14b`/`7b`) | ~$1.2–1.6 |
| g6.2xlarge | 1× L4 | 24 GB | Budget alt (lower throughput than A10G) | ~$1.0–1.5 |
| p4d/p5 | A100/H100 | 40–80 GB+ | High concurrency / larger models | $$$ |

- **EBS:** `gp3`, **300 GB** (models alone: qwen2.5:32b ≈ 20 GB, qwen2.5:7b ≈ 5 GB, HF models ≈ 5 GB; plus app data, Docker images, Weaviate/Neo4j data). 3000 IOPS / 125 MB/s default is fine.
- **vCPU/RAM:** the `.xlarge` sizes give 4 vCPU / 16–32 GB RAM — adequate for one user / light load. Use `.2xlarge` for more headroom.

> Pricing changes — confirm in the [AWS Pricing Calculator](https://calculator.aws). Sources for instance specs: [G6e](https://aws.amazon.com/ec2/instance-types/g6e/), [G5](https://aws.amazon.com/ec2/instance-types/g5/), [G6](https://aws.amazon.com/ec2/instance-types/g6/).

---

## 2. One-time AWS prerequisites

1. **GPU quota** — new accounts have a 0 vCPU limit on GPU families. In **Service Quotas → Amazon EC2 → "Running On-Demand G and VT instances"**, request enough vCPUs (g6e.xlarge = 4). Approval can take minutes to a day.
2. **Key pair** — EC2 → Key Pairs → create `futureverse-key` (.pem). `chmod 400 futureverse-key.pem`.
3. **Elastic IP** — allocate one (so the public IP survives reboots). You'll associate it after launch.
4. **Domain (recommended)** — a DNS A record you can point at the Elastic IP (Route 53 or any registrar). Needed for real TLS.
5. **Region** — pick one with GPU capacity near your users (e.g. `ap-south-1` Mumbai for India).

---

## 3. Step 1 — Launch the instance

EC2 → **Launch instance**:

- **Name:** `futureverse-prod`
- **AMI:** **Ubuntu Server 22.04 LTS** (x86_64). *(Alternative: "AWS Deep Learning Base GPU AMI (Ubuntu 22.04)" — ships with NVIDIA drivers + container toolkit preinstalled; skip parts of Step 2.)*
- **Instance type:** `g6e.xlarge` (or your choice from §1)
- **Key pair:** `futureverse-key`
- **Network / Security group — create new, inbound rules:**
  | Type | Port | Source |
  |---|---|---|
  | SSH | 22 | **My IP** (not 0.0.0.0/0) |
  | HTTP | 80 | 0.0.0.0/0 |
  | HTTPS | 443 | 0.0.0.0/0 |

  Do **not** open 8080/7687/8085/3306/6379 — they stay internal.
- **Storage:** 1× **gp3, 300 GiB**, delete-on-termination off if you want the data to survive.
- **(Optional) IAM instance profile:** a role with `AmazonS3FullAccess` (scoped to a backup bucket) + `CloudWatchAgentServerPolicy` if you'll do S3 backups / CloudWatch.
- **Launch**, then **associate the Elastic IP** with this instance.

---

## 4. Step 2 — Connect & prepare the OS

```bash
ssh -i futureverse-key.pem ubuntu@<ELASTIC_IP>

# --- NVIDIA driver (SKIP if using the Deep Learning AMI) ---
sudo apt-get update && sudo apt-get install -y build-essential
sudo apt-get install -y nvidia-driver-570          # 570+ supports CUDA 12.8
sudo reboot                                         # reconnect after ~30s
nvidia-smi                                          # must show the GPU

# --- Docker + Compose plugin ---
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && newgrp docker
docker version && docker compose version

# --- NVIDIA Container Toolkit (lets Docker use the GPU) ---
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# --- Verify Docker can see the GPU ---
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The last command **must** print the GPU. If it doesn't, fix this before continuing.

---

## 5. Step 3 — Get the code

```bash
sudo mkdir -p /opt/futureverse && sudo chown ubuntu:ubuntu /opt/futureverse
cd /opt/futureverse
# Private repo: use a deploy key or a fine-grained PAT
git clone https://github.com/futureverseai-in/futureverse-ai.git .
git checkout main          # after the `hardening` PR is merged
```

---

## 6. Step 4 — Reconcile the GPU build (important)

The Dockerfile installs `torch<=2.9.1` while `requirements.txt` pins `torch==2.11.0+cu128` (the Blackwell build) — they conflict, and you don't need cu128 on EC2 (A10G/L4/L40S are Ampere/Ada). Pick one **before building**:

```bash
# Easiest: make requirements match the Dockerfile's CUDA torch (works on all EC2 GPUs)
cd backend
sed -i 's#^torch==.*#torch==2.9.1#' requirements.txt
sed -i 's#^torchvision==.*#torchvision#; s#^torchaudio==.*#torchaudio#' requirements.txt
# (the Dockerfile installs the matching +cuXXX wheels from the PyTorch index for you)
cd ..
```

Build the image with CUDA + a non-root user + your embedding model baked in:

```bash
# (the docker-compose.prod.yaml below passes these build args; this is just to pre-pull base layers)
export DOCKER_BUILDKIT=1
```

---

## 7. Step 5 — Secrets & environment

```bash
cd /opt/futureverse

# 1) Compose secret files (gitignored)
mkdir -p secrets
openssl rand -hex 32                  > secrets/webui_secret_key
printf '%s' 'CHANGE_ME_neo4j_pass'    > secrets/neo4j_password
printf '%s' 'CHANGE_ME_weaviate_key'  > secrets/weaviate_api_key
chmod 600 secrets/*

# 2) Backend env
cp backend/.env.example backend/.env
nano backend/.env
```

Set these in `backend/.env` (internal hostnames match the compose service names):

```ini
ENV=prod
DEVICE_TYPE=cuda
VECTOR_DB=weaviate
WEAVIATE_HTTP_HOST=weaviate
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_HOST=weaviate
WEAVIATE_GRPC_PORT=50051
DATABASE_URL=mysql+pymysql://futureverseai:CHANGE_ME_mysql@mysql:3306/futureverseai
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
ENABLE_NEO4J=true
OLLAMA_BASE_URL=http://ollama:11434
RAG_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RAG_RERANKING_MODEL=BAAI/bge-reranker-v2-m3
ENABLE_RAG_HYBRID_SEARCH=true
CORS_ALLOW_ORIGIN=https://your.domain.com
ENABLE_SIGNUP=false
WEBUI_ADMIN_EMAIL=you@example.com
WEBUI_ADMIN_PASSWORD=CHANGE_ME_admin
# Also set MYSQL_PASSWORD / MYSQL_ROOT_PASSWORD / CORS_ALLOW_ORIGIN for the compose:
MYSQL_PASSWORD=CHANGE_ME_mysql
MYSQL_ROOT_PASSWORD=CHANGE_ME_root
WEAVIATE_API_KEY=CHANGE_ME_weaviate_key
```

> The compose `docker-compose.prod.yaml` reads `${MYSQL_PASSWORD}` etc. from `backend/.env` (or an env file you pass with `--env-file backend/.env`). Keep `CORS_ALLOW_ORIGIN` = your real HTTPS domain.

**MySQL tuning** (avoids the file-list `sort_buffer` error 1038): the compose `mysql` service should set `--sort-buffer-size=4M`. If you manage MySQL yourself, add `sort_buffer_size = 4M` to `my.cnf`.

---

## 8. Step 6 — Launch the stack

```bash
cd /opt/futureverse
# build + start everything (first build pulls base images + bakes embedding model: ~10–20 min)
docker compose -f docker-compose.prod.yaml --env-file backend/.env up -d --build

docker compose -f docker-compose.prod.yaml ps          # all should be healthy
docker compose -f docker-compose.prod.yaml logs -f backend   # watch startup + migrations
```

The backend runs Alembic migrations on startup. When you see `Application startup complete`, it's up.

**Pull the Ollama models** (into the ollama volume — one-time, large download):

```bash
docker compose -f docker-compose.prod.yaml exec ollama ollama pull qwen2.5:32b
docker compose -f docker-compose.prod.yaml exec ollama ollama pull qwen2.5:7b
docker compose -f docker-compose.prod.yaml exec ollama ollama ps     # confirm GPU use
```

**Verify health** (from the instance):

```bash
curl -fsS http://localhost:8080/health        # {"status": true}
curl -fsS http://localhost:8080/health/db
curl -fsS http://localhost:8080/health/ready  # per-dependency; 503 until all up
```

---

## 9. Step 7 — Reverse proxy + HTTPS + domain

Point your domain's **A record → Elastic IP** first. Then add **Caddy** (automatic Let's Encrypt TLS) as a front door.

`/opt/futureverse/Caddyfile`:
```
your.domain.com {
    reverse_proxy backend:8080
}
```

Add a Caddy service to the compose (or run standalone):
```bash
docker run -d --name caddy --restart unless-stopped \
  --network futureverse_default \
  -p 80:80 -p 443:443 \
  -v /opt/futureverse/Caddyfile:/etc/caddy/Caddyfile \
  -v caddy_data:/data \
  caddy:2
```
(Replace `futureverse_default` with the compose network name from `docker network ls`.)

Caddy fetches a real certificate automatically. Now browse **https://your.domain.com**. Make sure `CORS_ALLOW_ORIGIN` in `.env` equals this exact URL, then `docker compose ... up -d` to apply.

*(Nginx + certbot is the alternative if you prefer.)*

---

## 10. Step 8 — Production hardening

- **Elastic IP** associated (done) so the address is stable.
- **Non-root container:** build with `UID=1000 GID=1000` (compose `backend.build.args`), so the app doesn't run as root.
- **Secrets:** real values only in `secrets/` (gitignored) and `backend/.env` (gitignored) — never committed.
- **`WEBUI_SECRET_KEY`** is the file in `secrets/` — keep it stable so logins survive restarts; same value if you ever scale to >1 instance.
- **Rotate** `WEBUI_ADMIN_PASSWORD` after first login; set `ENABLE_SIGNUP=false`.
- **Security group** stays minimal (22 from your IP, 80/443 public). Consider AWS SSM Session Manager instead of SSH to close port 22 entirely.
- **OS updates:** `sudo unattended-upgrades` for security patches.

---

## 11. Step 9 — Persistence, backups, monitoring

**Persistence:** all state is in Docker named volumes (`backend_data`, `weaviate_data`, `neo4j_data`, `mysql_data`, `redis_data`, `ollama_data`) under `/var/lib/docker/volumes`. For safety, keep that on the EBS root volume (or mount a separate gp3 data volume at `/var/lib/docker`).

**Backups:**
```bash
# MySQL
docker compose -f docker-compose.prod.yaml exec mysql \
  sh -c 'exec mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" futureverseai' > backup-mysql.sql
# Neo4j (offline dump)
docker compose -f docker-compose.prod.yaml exec neo4j \
  neo4j-admin database dump neo4j --to-path=/data/backups
# Weaviate: snapshot the volume, or enable the backup module to S3
```
Automate with **AWS Data Lifecycle Manager** (daily EBS snapshots) and/or a cron that pushes the dumps to an **S3 bucket** (via the IAM role).

**Monitoring:**
- `docker compose ps` / `docker stats`, and the `/health/ready` probe (wire it to an uptime check).
- GPU: `nvidia-smi` or install **DCGM exporter** + CloudWatch.
- App logs: `docker compose logs`; ship to CloudWatch Logs with the CloudWatch agent if desired.
- **Restart on boot:** services use `restart: unless-stopped`, and `sudo systemctl enable docker` ensures Docker starts at boot.

---

## 12. Step 10 — Updates / redeploy

```bash
cd /opt/futureverse
git pull
docker compose -f docker-compose.prod.yaml --env-file backend/.env up -d --build
# migrations run on backend startup; check logs
docker compose -f docker-compose.prod.yaml logs -f backend
```

---

## 13. Optional — managed-services split (for scale)

Keep the GPU instance for **Ollama + embeddings + Weaviate + Neo4j + backend**, but move stateful infra to managed services:
- **Amazon RDS for MySQL 8** → point `DATABASE_URL` at the RDS endpoint (remove the `mysql` compose service).
- **Amazon ElastiCache (Redis)** → point `REDIS_URL` at it.
- **Application Load Balancer + ACM cert** in front (replaces Caddy) if you run multiple backend instances — but the GPU services (Ollama, embedder) still pin to the GPU host.
- Weaviate/Neo4j can move to dedicated instances or managed Aura/WCS if you outgrow one box.

---

## 14. Cost & teardown

- **Running cost** is dominated by the GPU instance (~$1.5–2.5/hr for g6e.xlarge on-demand) + EBS (~$0.08/GB-mo × 300 ≈ $24/mo) + Elastic IP (free while attached) + data transfer out. A **1-year Savings Plan** cuts the instance ~40%. **Stop the instance when idle** to pause GPU charges (EBS still bills).
- **Teardown to stop all charges:**
  ```bash
  docker compose -f docker-compose.prod.yaml down -v   # stop + delete volumes (data!)
  ```
  then in the console: **terminate** the instance, **delete** the EBS volume(s), **release** the Elastic IP.

---

## 15. Troubleshooting

| Symptom | Fix |
|---|---|
| `docker run --gpus all` can't see GPU | Re-run the nvidia-container-toolkit steps; `sudo systemctl restart docker`; confirm `nvidia-smi` on the host. |
| Ollama runs on CPU (slow) | On EC2 it should use CUDA automatically; check `ollama ps`. Ensure the `ollama` service has the GPU reservation and the toolkit is installed. |
| CUDA out of memory | Answer model too big for the GPU — use `qwen2.5:14b`/`7b`, or move to g6e (48 GB). `OLLAMA_MAX_LOADED_MODELS=1` keeps one LLM resident. |
| Knowledge base shows "No content found" | Ensure you're on the patched build (MySQL `sort_buffer` 1038 fix) and MySQL has `sort_buffer_size=4M`. |
| Files added to a KB have ≤25 chunks | Patched build copies all chunks; `QUERY_DEFAULTS_LIMIT` in `weaviate/docker-compose.yml` only affects search defaults. Re-add affected files. |
| TLS cert fails | DNS A record must point at the Elastic IP and ports 80/443 must be open before Caddy can validate. |
| `ModuleNotFoundError: futureverse_ai` | The image launches from `backend/`; if running manually, `cd backend` first. |

---

*Companion docs: [`../SETUP.md`](../SETUP.md) (local run/config), [`RAG_PIPELINE.md`](RAG_PIPELINE.md) (architecture).*
