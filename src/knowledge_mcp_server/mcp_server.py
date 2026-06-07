"""Minimal MCP-style server for agents to add and retrieve markdown docs.

Endpoints:
- POST /mcp/documents          Add a markdown document (repo_name optional)
- GET  /mcp/documents          List documents metadata
- GET  /mcp/documents/{doc_id} Get raw document text
- POST /mcp/query              Query RAG and return top-k results
- POST /mcp/index_url          Index a website URL
- POST /mcp/index_pdf          Index a PDF file (upload)
- POST /mcp/index_tex          Index a TeX file (upload)

This server uses the package's LocalTextStore, Embeddings, and InMemoryVectorStore.
"""

import logging
import tempfile
import uuid
from contextlib import suppress
from os import getenv
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from knowledge_mcp_server.embeddings import Embeddings
from knowledge_mcp_server.indexers import index_pdf, index_tex, index_website
from knowledge_mcp_server.storage import LocalTextStore
from knowledge_mcp_server.vectorstore import InMemoryVectorStore

logger = logging.getLogger(__name__)

_WEB_UI = Path(__file__).parent.parent.parent / "web_ui"

app = FastAPI(title="knowledge_mcp_server MCP")

# initialize components (in-memory) and try to load persisted vectorstore
store = LocalTextStore()
emb = Embeddings()
try:
    vecstore = InMemoryVectorStore.load(root=store.root)
    logger.info("Loaded existing vector store")
except (FileNotFoundError, OSError):
    vecstore = InMemoryVectorStore()
    logger.info("Created new vector store")

# mount static web UI under /static and serve index.html at root
app.mount("/static", StaticFiles(directory=str(_WEB_UI)), name="static")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(str(_WEB_UI / "index.html"))


class AddDocReq(BaseModel):
    repo: str | None = None
    title: str | None = None
    markdown: str


class QueryReq(BaseModel):
    q: str
    top_k: int = 5


@app.post("/mcp/documents")
def add_document(req: AddDocReq) -> dict[str, str | int]:
    if not req.markdown or not req.markdown.strip():
        raise HTTPException(status_code=400, detail="Empty markdown")
    # create a doc id
    base = (req.title or req.repo or "doc").replace(" ", "_")[:40]
    # simple id; store will also record metadata
    doc_id = f"{base}-{uuid.uuid4().hex[:8]}"
    metadata = {"repo": req.repo, "title": req.title}
    store.add_doc(doc_id, req.markdown, metadata=metadata)
    # index into vectorstore by simple chunking using embeddings
    # naive: split by paragraphs
    parts = [p.strip() for p in req.markdown.split("\n\n") if p.strip()]
    if parts:
        vecs = emb.embed(parts)
        ids = [f"{doc_id}-{i}" for i in range(len(parts))]
        metadatas = [{"doc_id": doc_id, "part_index": i} for i in range(len(parts))]
        vecstore.add(ids, vecs, metadatas)
        # persist vectorstore after adding
        with suppress(OSError):
            vecstore.save(root=store.root)
    return {"doc_id": doc_id, "indexed_parts": len(parts)}


class IndexUrlReq(BaseModel):
    url: str
    repo: str | None = None
    title: str | None = None


@app.post("/mcp/index_url")
def index_url(req: IndexUrlReq) -> dict[str, list[str] | int]:
    if not req.url:
        raise HTTPException(status_code=400, detail="Missing url")
    # use indexers.index_website to fetch, chunk and index
    try:
        ids = index_website(req.url, store, emb, vecstore)
        with suppress(OSError):
            vecstore.save(root=store.root)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"indexed_ids": ids, "count": len(ids)}


@app.post("/mcp/index_pdf")
async def index_pdf_upload(file: UploadFile = File(...)) -> dict[str, list[str] | int | str]:
    """Upload and index a PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Index the PDF
        ids = index_pdf(tmp_path, store, emb, vecstore)

        # Clean up temp file
        Path(tmp_path).unlink()

        # Save vector store
        with suppress(OSError):
            vecstore.save(root=store.root)

        return {"indexed_ids": ids, "count": len(ids), "filename": file.filename}
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/mcp/index_tex")
async def index_tex_upload(file: UploadFile = File(...)) -> dict[str, list[str] | int | str]:
    """Upload and index a TeX file."""
    if not (file.filename.endswith(".tex") or file.filename.endswith(".latex")):
        raise HTTPException(
            status_code=400,
            detail="File must be a .tex or .latex file",
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tex", mode="wb") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Index the TeX file
        ids = index_tex(tmp_path, store, emb, vecstore)

        # Clean up temp file
        Path(tmp_path).unlink()

        # Save vector store
        with suppress(OSError):
            vecstore.save(root=store.root)

        return {"indexed_ids": ids, "count": len(ids), "filename": file.filename}
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/mcp/documents")
def list_documents() -> dict[str, dict]:
    return store.list_docs()


@app.get("/mcp/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, str]:
    txt = store.get_doc(doc_id)
    if txt is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "text": txt}


@app.post("/mcp/query")
def query(req: QueryReq) -> dict[str, list[dict]]:
    qv = emb.embed([req.q])[0]
    results = vecstore.search(qv, top_k=req.top_k)
    out = []
    for rid, score, meta in results:
        # try several fallbacks to retrieve the stored chunk text for better responses
        text = None
        # 1) metadata may include a doc_id that points to a stored document
        if isinstance(meta, dict):
            doc_id = meta.get("doc_id")
            if doc_id:
                text = store.get_doc(doc_id)
            # 2) metadata might include a direct path to the saved chunk file
            if text is None and meta.get("path"):
                try:
                    text = Path(meta["path"]).read_text(encoding="utf-8")
                except (OSError, KeyError):
                    text = None
        # 3) fallback: the search result id itself is often the chunk id stored in LocalTextStore
        if text is None:
            text = store.get_doc(rid)
        # limit returned text size to keep responses compact
        snippet = text[:2000] if text else None
        out.append({"id": rid, "score": score, "meta": meta, "doc_text": snippet})
    return {"results": out}


@app.api_route(
    "/.well-known/mcp.json",
    methods=["GET", "POST"],
    include_in_schema=False,
)
def well_known_manifest() -> dict:
    """Return a minimal MCP manifest describing the server's tools.

    This manifest advertises no authentication so it matches the server's
    current behavior (no API-key enforcement). If you later enable API-key
    enforcement, update this manifest accordingly.
    """
    base = getenv("MCP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    # Accept both GET and POST to be tolerant of clients that POST the manifest URL.
    return {
        "name": "DevKnowledge MCP (local)",
        "version": "0.1",
        "description": "Minimal MCP manifest for the local RAG/MCP server",
        "api": {
            "post_doc": f"{base}/mcp/documents",
            "list_docs": f"{base}/mcp/documents",
            "get_doc": f"{base}/mcp/documents/{{doc_id}}",
            "query": f"{base}/mcp/query",
            "index_url": f"{base}/mcp/index_url",
            "index_pdf": f"{base}/mcp/index_pdf",
            "index_tex": f"{base}/mcp/index_tex",
        },
        "auth": {"type": "none"},
        "tools": [
            {
                "name": "post_doc",
                "description": "Add a markdown document via POST /mcp/documents",
            },
            {
                "name": "list_docs",
                "description": "List stored documents via GET /mcp/documents",
            },
            {
                "name": "get_doc",
                "description": "Get raw document text via GET /mcp/documents/{doc_id}",
            },
            {
                "name": "query",
                "description": "Query the vectorstore via POST /mcp/query",
            },
            {
                "name": "index_url",
                "description": "Fetch and index a URL via POST /mcp/index_url",
            },
            {
                "name": "index_pdf",
                "description": "Upload and index a PDF file via POST /mcp/index_pdf",
            },
            {
                "name": "index_tex",
                "description": "Upload and index a TeX file via POST /mcp/index_tex",
            },
        ],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
