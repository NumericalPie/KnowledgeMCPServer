"""MCP stdio server implementation for VS Code integration.

This implements the Model Context Protocol over stdio (JSON-RPC) for proper
VS Code/GitHub Copilot integration.
"""

from __future__ import annotations

import asyncio
import logging  # ADR-0003 exemption: loguru uses stderr, corrupting the stdio JSON-RPC stream
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from knowledge_mcp_server.embeddings import Embeddings
from knowledge_mcp_server.indexers import index_website
from knowledge_mcp_server.storage import LocalTextStore
from knowledge_mcp_server.vectorstore import InMemoryVectorStore

# Set up logging to file to avoid interfering with stdio
logging.basicConfig(
    level=logging.INFO,
    filename=str(Path(tempfile.gettempdir()) / "mcp_server.log"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_stdio_server")

# Initialize components
store = LocalTextStore()
emb = Embeddings()
try:
    vecstore = InMemoryVectorStore.load(root=store.root)
    logger.info("Loaded existing vector store")
except (FileNotFoundError, OSError) as e:
    vecstore = InMemoryVectorStore()
    logger.info(f"Created new vector store: {e}")

# Create MCP server
app = Server("knowledge-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="query_knowledge",
            description=(
                "Search the knowledge base using semantic search. "
                "Returns relevant document chunks with similarity scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {
                        "type": "number",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="index_url",
            description="Fetch and index content from a URL into the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to index"},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="add_document",
            description="Add a markdown document to the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "markdown": {
                        "type": "string",
                        "description": "The markdown content to add",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the document",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Optional repository name",
                    },
                },
                "required": ["markdown"],
            },
        ),
        Tool(
            name="list_documents",
            description="List all documents in the knowledge base",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def _query_knowledge(arguments: dict) -> list[TextContent]:
    """Execute query_knowledge tool."""
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)

    if not query:
        return [TextContent(type="text", text="Error: Empty query")]

    qv = emb.embed([query])[0]
    results = vecstore.search(qv, top_k=int(top_k))

    if not results:
        return [TextContent(type="text", text="No results found")]

    response_parts = []
    for rid, score, meta in results:
        # Try to retrieve the stored chunk text
        text = None
        if isinstance(meta, dict):
            doc_id = meta.get("doc_id")
            if doc_id:
                text = store.get_doc(doc_id)
            if text is None and meta.get("path"):
                with suppress(OSError, KeyError):
                    text = Path(meta["path"]).read_text(encoding="utf-8")
        if text is None:
            text = store.get_doc(rid)

        snippet = text[:1000] if text else "(no text available)"
        source = meta.get("source", rid) if isinstance(meta, dict) else rid
        response_parts.append(f"[Score: {score:.3f}] {source}\n{snippet}\n")

    return [TextContent(type="text", text="\n---\n".join(response_parts))]


def _index_url(arguments: dict) -> list[TextContent]:
    """Execute index_url tool."""
    url = arguments.get("url", "")
    if not url:
        return [TextContent(type="text", text="Error: Missing URL")]

    try:
        ids = index_website(url, store, emb, vecstore)
        vecstore.save(root=store.root)
        return [
            TextContent(
                type="text",
                text=f"Successfully indexed {len(ids)} chunks from {url}",
            ),
        ]
    except (OSError, ValueError) as e:
        return [TextContent(type="text", text=f"Error indexing URL: {e!s}")]


def _add_document(arguments: dict) -> list[TextContent]:
    """Execute add_document tool."""
    markdown = arguments.get("markdown", "")
    title = arguments.get("title")
    repo = arguments.get("repo")

    if not markdown.strip():
        return [TextContent(type="text", text="Error: Empty markdown")]

    base = (title or repo or "doc").replace(" ", "_")[:40]
    doc_id = f"{base}-{uuid.uuid4().hex[:8]}"
    metadata = {"repo": repo, "title": title}
    store.add_doc(doc_id, markdown, metadata=metadata)

    # Index into vectorstore
    parts = [p.strip() for p in markdown.split("\n\n") if p.strip()]
    if parts:
        vecs = emb.embed(parts)
        ids = [f"{doc_id}-{i}" for i in range(len(parts))]
        metadatas = [{"doc_id": doc_id, "part_index": i} for i in range(len(parts))]
        vecstore.add(ids, vecs, metadatas)
        vecstore.save(root=store.root)

    return [
        TextContent(
            type="text",
            text=f"Added document {doc_id} with {len(parts)} indexed parts",
        ),
    ]


def _list_documents(_arguments: dict) -> list[TextContent]:
    """Execute list_documents tool."""
    docs = store.list_docs()
    if not docs:
        return [TextContent(type="text", text="No documents found")]

    doc_list = []
    for doc_id, metadata in docs.items():
        title = metadata.get("title", doc_id)
        repo = metadata.get("repo", "")
        doc_list.append(f"- {doc_id}: {title} ({repo})")

    return [TextContent(type="text", text="\n".join(doc_list))]


# Tool dispatch table
_TOOL_HANDLERS = {
    "query_knowledge": _query_knowledge,
    "index_url": _index_url,
    "add_document": _add_document,
    "list_documents": _list_documents,
}


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution via dispatch."""
    try:
        handler = _TOOL_HANDLERS.get(name)
        if handler:
            return handler(arguments)
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except (OSError, ValueError, KeyError) as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {e!s}")]


async def main() -> None:
    """Run the MCP server via stdio."""
    logger.info("Starting MCP stdio server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
