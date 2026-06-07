"""Tests for the FastAPI MCP server endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from knowledge_mcp_server.mcp_server import app


@pytest.fixture
def client():
    """Return a TestClient with mocked store, emb, and vecstore."""
    mock_store = MagicMock()
    mock_store.root = "test_store"
    mock_store.add_doc = MagicMock()
    mock_store.get_doc = MagicMock(return_value=None)
    mock_store.list_docs = MagicMock(return_value={})

    mock_emb = MagicMock()
    mock_emb.embed = MagicMock(
        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )

    mock_vec = MagicMock()
    mock_vec.search = MagicMock(return_value=[])
    mock_vec.add = MagicMock()
    mock_vec.save = MagicMock()

    with (
        patch("knowledge_mcp_server.mcp_server.store", mock_store),
        patch("knowledge_mcp_server.mcp_server.emb", mock_emb),
        patch("knowledge_mcp_server.mcp_server.vecstore", mock_vec),
    ):
        yield TestClient(app), mock_store, mock_emb, mock_vec


class TestAddDocument:
    def test_add_document_success(self, client):
        tc, mock_store, mock_emb, _mock_vec = client
        mock_emb.embed.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        resp = tc.post("/mcp/documents", json={"markdown": "# Hello\n\nWorld"})

        assert resp.status_code == 200
        data = resp.json()
        assert "doc_id" in data
        assert data["indexed_parts"] >= 1
        mock_store.add_doc.assert_called_once()

    def test_add_document_empty_rejects(self, client):
        tc, *_ = client
        resp = tc.post("/mcp/documents", json={"markdown": "   "})
        assert resp.status_code == 400

    def test_add_document_with_title_and_repo(self, client):
        tc, mock_store, mock_emb, _ = client
        mock_emb.embed.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        resp = tc.post(
            "/mcp/documents",
            json={"markdown": "content", "title": "My Doc", "repo": "my-repo"},
        )

        assert resp.status_code == 200
        call_args = mock_store.add_doc.call_args
        assert call_args[1]["metadata"]["title"] == "My Doc"
        assert call_args[1]["metadata"]["repo"] == "my-repo"


class TestListDocuments:
    def test_list_returns_metadata(self, client):
        tc, mock_store, *_ = client
        mock_store.list_docs.return_value = {
            "doc-abc": {"title": "T", "path": "x.txt"}
        }

        resp = tc.get("/mcp/documents")

        assert resp.status_code == 200
        assert "doc-abc" in resp.json()

    def test_list_empty_store(self, client):
        tc, *_ = client
        resp = tc.get("/mcp/documents")
        assert resp.status_code == 200
        assert resp.json() == {}


class TestGetDocument:
    def test_get_existing_document(self, client):
        tc, mock_store, *_ = client
        mock_store.get_doc.return_value = "hello world"

        resp = tc.get("/mcp/documents/my-doc")

        assert resp.status_code == 200
        assert resp.json()["text"] == "hello world"

    def test_get_missing_document_returns_404(self, client):
        tc, mock_store, *_ = client
        mock_store.get_doc.return_value = None

        resp = tc.get("/mcp/documents/missing")

        assert resp.status_code == 404


class TestQuery:
    def test_query_returns_results(self, client):
        tc, mock_store, mock_emb, mock_vec = client
        mock_emb.embed.return_value = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        mock_vec.search.return_value = [("chunk-1", 0.95, {"doc_id": "doc-1"})]
        mock_store.get_doc.return_value = "Some chunk text"

        resp = tc.post("/mcp/query", json={"q": "find something", "top_k": 3})

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.95)
        assert results[0]["doc_text"] == "Some chunk text"

    def test_query_empty_vectorstore(self, client):
        tc, _, mock_emb, mock_vec = client
        mock_emb.embed.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        mock_vec.search.return_value = []

        resp = tc.post("/mcp/query", json={"q": "anything"})

        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_query_truncates_long_text(self, client):
        tc, mock_store, mock_emb, mock_vec = client
        mock_emb.embed.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        mock_vec.search.return_value = [("chunk-1", 0.9, {"doc_id": "doc-1"})]
        mock_store.get_doc.return_value = "x" * 5000

        resp = tc.post("/mcp/query", json={"q": "test"})

        assert len(resp.json()["results"][0]["doc_text"]) == 2000


class TestIndexUrl:
    @patch("knowledge_mcp_server.mcp_server.index_website")
    def test_index_url_success(self, mock_index, client):
        tc, *_ = client
        mock_index.return_value = ["id1", "id2"]

        resp = tc.post("/mcp/index_url", json={"url": "https://example.com"})

        assert resp.status_code == 200
        assert resp.json()["count"] == 2
        mock_index.assert_called_once()

    def test_index_url_missing_url(self, client):
        tc, *_ = client
        resp = tc.post("/mcp/index_url", json={"url": ""})
        assert resp.status_code == 400

    @patch("knowledge_mcp_server.mcp_server.index_website")
    def test_index_url_propagates_errors(self, mock_index, client):
        tc, *_ = client
        mock_index.side_effect = OSError("network fail")

        resp = tc.post("/mcp/index_url", json={"url": "https://example.com"})

        assert resp.status_code == 500


class TestWellKnownManifest:
    def test_manifest_get(self, client):
        tc, *_ = client
        resp = tc.get("/.well-known/mcp.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert data["auth"]["type"] == "none"

    def test_manifest_post(self, client):
        tc, *_ = client
        resp = tc.post("/.well-known/mcp.json")
        assert resp.status_code == 200
