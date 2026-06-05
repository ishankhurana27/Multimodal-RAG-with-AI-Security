# Contributing Guide

## Branch Strategy

```
main        ← production, protected, requires 2 approvals
develop     ← integration branch, requires 1 approval
feature/    ← new features, branch from develop
fix/        ← bug fixes, branch from develop
chore/      ← deps, tooling, config
hotfix/     ← urgent production fixes, branch from main
```

## Commit Messages — Conventional Commits

Format: `<type>(<scope>): <summary>`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`, `ci`

Scopes:
```
backend, frontend, auth, chat, knowledge, retrieval,
graph, vector, models, admin, api, migrations, docker, ci
```

Examples:
```
feat(knowledge): add RAG chunking strategy selector
fix(auth): resolve JWT expiry not refreshing session
chore(migrations): add index on question_behavioral_signals
refactor(retrieval): extract graph pipeline into service layer
ci: add path filters to frontend workflow
```

## Pull Requests

1. Branch from `develop` (or `main` for hotfix)
2. Open PR against `develop`
3. Fill in the PR template completely
4. CI must pass before requesting review
5. 1 approval required to merge into `develop`
6. 2 approvals required to merge into `main`
7. Delete branch after merge

## Database Rule

Every change to a SQLAlchemy model MUST have a matching
Alembic migration. Never change a model without running:
```bash
alembic revision --autogenerate -m "what changed"
alembic upgrade head
```

Migrations in `backend/futureverse_ai/migrations/versions/`
are source code — always commit them.

## Python Standards

- Formatter: `black` (line length 100)
- Linter: `ruff`
- Type hints on all public functions
- No `print()` in committed code — use the logger

## TypeScript/Svelte Standards

- ESLint + Prettier
- No `any` types without an explaining comment
- One component per file

## What NOT to Commit

- `.env` files (any depth)
- `backend/data/` — user uploads, SQLite, cache
- `backend/qdrant_storage/` — vector DB data
- `backend/futureverse_ai/static/` — build artifact, run `npm run build`
- `node_modules/`
- Any file with real API keys, passwords, or tokens
