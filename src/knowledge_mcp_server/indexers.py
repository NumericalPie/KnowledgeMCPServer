import re
import uuid
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

from .embeddings import Embeddings
from .storage import LocalTextStore
from .vectorstore import InMemoryVectorStore


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks: list[str] = []

    if chunk_size <= 0:
        err_msg = "chunk_size must be a positive integer"
        raise ValueError(err_msg)

    # empty input -> return single empty chunk (tests expect this)
    if not words:
        return [""]

    # if the whole text fits within one chunk, return it as a single chunk
    if len(words) <= chunk_size:
        return [" ".join(words)]

    # ensure overlap is non-negative
    overlap = max(0, overlap)
    # compute step and guard against non-positive step
    step = chunk_size - overlap
    if step <= 0:
        # If overlap >= chunk_size, treat as minimal forward step of 1 word
        step = 1

    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += step

    return chunks


def index_pdf(
    path: str,
    store: LocalTextStore,
    embeddings: Embeddings,
    vectorstore: InMemoryVectorStore,
    chunk_size: int = 800,
) -> list[str]:
    text = extract_text(path)
    base_id = Path(path).stem
    chunks = _chunk_text(text, chunk_size=chunk_size)
    ids = []
    metadatas = []
    for i, c in enumerate(chunks):
        doc_id = f"{base_id}-{i}-{uuid.uuid4().hex[:6]}"
        ids.append(doc_id)
        metadatas.append({"source": path, "chunk_index": i})
        store.add_doc(doc_id, c, metadata=metadatas[-1])
    vecs = embeddings.embed(chunks)
    vectorstore.add(ids, vecs, metadatas)
    return ids


def index_tex(
    path: str,
    store: LocalTextStore,
    embeddings: Embeddings,
    vectorstore: InMemoryVectorStore,
    chunk_size: int = 800,
) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    # very naive strip of LaTeX commands
    text = re.sub(r"\\\\[a-zA-Z]+\{.*?\}", "", raw)
    text = re.sub(r"\\\\[a-zA-Z]+", "", text)
    base_id = Path(path).stem
    chunks = _chunk_text(text, chunk_size=chunk_size)
    ids = []
    metadatas = []
    for i, c in enumerate(chunks):
        doc_id = f"{base_id}-{i}-{uuid.uuid4().hex[:6]}"
        ids.append(doc_id)
        metadatas.append({"source": path, "chunk_index": i})
        store.add_doc(doc_id, c, metadata=metadatas[-1])
    vecs = embeddings.embed(chunks)
    vectorstore.add(ids, vecs, metadatas)
    return ids


def index_website(
    url: str,
    store: LocalTextStore,
    embeddings: Embeddings,
    vectorstore: InMemoryVectorStore,
    chunk_size: int = 800,
) -> list[str]:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(separator=" ", strip=True)
    base_id = re.sub(r"[^a-zA-Z0-9]", "-", url)[:40]
    chunks = _chunk_text(text, chunk_size=chunk_size)
    ids = []
    metadatas = []
    for i, c in enumerate(chunks):
        doc_id = f"{base_id}-{i}-{uuid.uuid4().hex[:6]}"
        ids.append(doc_id)
        metadatas.append({"source": url, "chunk_index": i})
        store.add_doc(doc_id, c, metadata=metadatas[-1])
    vecs = embeddings.embed(chunks)
    vectorstore.add(ids, vecs, metadatas)
    return ids
