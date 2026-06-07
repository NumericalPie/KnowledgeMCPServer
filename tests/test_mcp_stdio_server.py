"""Tests for the stdio MCP server tool handlers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mcp.types import TextContent


@pytest.fixture(autouse=True)
def _patch_globals(tmp_path):
    """Patch module-level store, emb, and vecstore before import."""
    mock_store = MagicMock()
    mock_store.root = str(tmp_path)
    mock_emb = MagicMock()
    mock_emb.embed.return_value = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    mock_vec = MagicMock()
    mock_vec.search.return_value = []

    with (
        patch("knowledge_mcp_server.mcp_stdio_server.store", mock_store),
        patch("knowledge_mcp_server.mcp_stdio_server.emb", mock_emb),
        patch("knowledge_mcp_server.mcp_stdio_server.vecstore", mock_vec),
    ):
        yield mock_store, mock_emb, mock_vec


def _store(fixtures):
    return fixtures[0]


def _emb(fixtures):
    return fixtures[1]


def _vec(fixtures):
    return fixtures[2]


class TestQueryKnowledge:
    def test_empty_query_returns_error(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _query_knowledge

        result = _query_knowledge({"query": ""})

        assert isinstance(result[0], TextContent)
        assert "Error" in result[0].text

    def test_no_results(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _query_knowledge

        _vec(_patch_globals).search.return_value = []

        result = _query_knowledge({"query": "something"})

        assert "No results" in result[0].text

    def test_returns_formatted_results(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _query_knowledge

        _store(_patch_globals).get_doc.return_value = "chunk text"
        _vec(_patch_globals).search.return_value = [
            ("chunk-1", 0.92, {"doc_id": "doc-1", "source": "test.pdf"})
        ]

        result = _query_knowledge({"query": "find", "top_k": 1})

        assert "0.920" in result[0].text
        assert "chunk text" in result[0].text

    def test_top_k_passed_to_search(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _query_knowledge

        _vec(_patch_globals).search.return_value = []

        _query_knowledge({"query": "test", "top_k": 7})

        _vec(_patch_globals).search.assert_called_once()
        call_kwargs = _vec(_patch_globals).search.call_args
        assert call_kwargs[1].get("top_k") == 7 or call_kwargs[0][1] == 7


class TestIndexUrl:
    @patch("knowledge_mcp_server.mcp_stdio_server.index_website")
    def test_success(self, mock_index, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _index_url

        mock_index.return_value = ["id1", "id2", "id3"]

        result = _index_url({"url": "https://example.com"})

        assert "3" in result[0].text
        mock_index.assert_called_once()

    def test_missing_url_returns_error(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _index_url

        result = _index_url({"url": ""})

        assert "Error" in result[0].text

    @patch("knowledge_mcp_server.mcp_stdio_server.index_website")
    def test_handles_exception(self, mock_index, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _index_url

        mock_index.side_effect = OSError("connection refused")

        result = _index_url({"url": "https://example.com"})

        assert "Error" in result[0].text


class TestAddDocument:
    def test_adds_and_indexes_document(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _add_document

        _emb(_patch_globals).embed.return_value = np.array(
            [[1.0, 0.0]], dtype=np.float32
        )

        result = _add_document({"markdown": "# Title\n\nSome content here."})

        assert isinstance(result[0], TextContent)
        _store(_patch_globals).add_doc.assert_called_once()
        _vec(_patch_globals).add.assert_called_once()

    def test_empty_markdown_returns_error(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _add_document

        result = _add_document({"markdown": "   "})

        assert "Error" in result[0].text

    def test_uses_title_in_doc_id(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _add_document

        _emb(_patch_globals).embed.return_value = np.array(
            [[1.0, 0.0]], dtype=np.float32
        )

        result = _add_document({"markdown": "content", "title": "My Guide"})

        assert "My_Guide" in result[0].text


class TestListDocuments:
    def test_empty_store(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _list_documents

        _store(_patch_globals).list_docs.return_value = {}

        result = _list_documents({})

        assert "No documents" in result[0].text

    def test_lists_all_docs(self, _patch_globals):
        from knowledge_mcp_server.mcp_stdio_server import _list_documents

        _store(_patch_globals).list_docs.return_value = {
            "doc-1": {"title": "Guide A", "repo": "repo-x"},
            "doc-2": {"title": "Guide B", "repo": ""},
        }

        result = _list_documents({})

        assert "doc-1" in result[0].text
        assert "doc-2" in result[0].text
