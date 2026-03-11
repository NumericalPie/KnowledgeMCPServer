# KnowledgeMCPServer

Small, forkable RAG project with an MCP server (for VS Code/Copilot) and a minimal web UI.

## Quick Start

Prerequisite: install `uv` from https://docs.astral.sh/uv/

```bash
uv sync
source .venv/bin/activate
```

Run tests:

```bash
uv run pytest
```

## Use It With Your Own Knowledge Set

1. Start from a clean data state (optional but recommended):

```bash
rm -f data/docs/* data/vectorstore/*
echo '{}' > data/metadata.json
```

2. Index content:

```bash
uv run python -m knowledge_mcp_server.cli index --url https://example.com
# or --pdf /path/to/file.pdf
# or --tex /path/to/file.tex
```

3. Query via MCP or HTTP/web UI.

## Register MCP Server In VS Code

```bash
code --add-mcp '{"name":"knowledge-mcp-server","command":"uv","args":["run","python","-m","knowledge_mcp_server.mcp_stdio_server"],"cwd":"'$(pwd)'"}'
```

## Run Web UI

```bash
uv run uvicorn knowledge_mcp_server.mcp_server:app --reload --port 8000
```

Open `http://127.0.0.1:8000/`.

## Project Layout

- Code: `src/knowledge_mcp_server/`
- Tests: `tests/`
- Runtime data: `data/docs/`, `data/vectorstore/`, `data/metadata.json`

## Defaults

- Embeddings: `all-MiniLM-L6-v2`
- Chunking: 800 words, 100 overlap
- HTTP port: `8000`

## Docs

- `docs/MCP_SETUP.md` - MCP registration and troubleshooting
- `docs/ARCHITECTURE.md` - architecture notes

## License

MIT (see `LICENSE`)
