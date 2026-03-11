# KnowledgeMCPServer — Architecture

Components
- MCP stdio server: `mcp_stdio_server.py` (stdio JSON-RPC for Copilot Chat)
- HTTP server: `mcp_server.py` (FastAPI web UI + REST endpoints)
- Storage: `LocalTextStore` (files under `data/docs/` + `metadata.json`)
- Vector store: `InMemoryVectorStore` (vectors persisted to `data/vectorstore/`)
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2` default)
- Indexers: `indexers.py` (PDF, TeX, website)

Data flow
- Index: Content → Indexer → Chunker (default 800/100) → Embeddings → VectorStore → Disk
- Query: Query → Embeddings → Vector search → Top-K results (with metadata)

Storage layout
- `data/docs/` : chunk files
- `data/metadata.json` : registry
- `data/vectorstore/` : `ids.json`, `vectors.npy`, `metadatas.json`

APIs (quick)
- MCP tools: `query_knowledge`, `index_url`, `add_document`, `list_documents`
- HTTP: `POST /mcp/query`, `POST /mcp/index_url`, `POST /mcp/add_document`, `GET /mcp/documents`

Defaults
- Embeddings model: `all-MiniLM-L6-v2`
- HTTP port: `8000`
- Chunk size/overlap: `800/100`
- Storage root: `data/`

This file lists the minimal architecturally-relevant details for maintainers.