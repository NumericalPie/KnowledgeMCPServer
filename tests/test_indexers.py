"""Tests for indexers module."""

from unittest.mock import Mock, patch

import pytest

from knowledge_mcp_server.indexers import (
    _chunk_text,
    index_pdf,
    index_tex,
    index_website,
)


class TestChunkText:
    """Test suite for _chunk_text function."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = _chunk_text(text, chunk_size=5, overlap=1)

        assert len(chunks) > 1
        # Each chunk should have roughly 5 words
        assert len(chunks[0].split()) <= 5

    def test_chunk_text_with_overlap(self):
        """Test chunking maintains overlap."""
        text = "w1 w2 w3 w4 w5 w6 w7 w8"
        chunks = _chunk_text(text, chunk_size=4, overlap=2)

        # Verify overlap exists
        assert "w3" in chunks[0] or "w4" in chunks[0]
        assert "w3" in chunks[1] or "w4" in chunks[1]

    def test_chunk_text_short_text(self):
        """Test chunking text shorter than chunk_size."""
        text = "short text"
        chunks = _chunk_text(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = _chunk_text("", chunk_size=10)
        assert len(chunks) == 1
        assert chunks[0] == ""


class TestIndexPdf:
    """Test suite for index_pdf function."""

    @patch("knowledge_mcp_server.indexers.extract_text")
    def test_index_pdf_creates_chunks(
        self, mock_extract, temp_store, mock_embeddings, empty_vectorstore, tmp_path
    ):
        """Test PDF indexing creates chunks."""
        mock_extract.return_value = "This is a test PDF content with multiple words."
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        ids = index_pdf(str(pdf_path), temp_store, mock_embeddings, empty_vectorstore)

        assert len(ids) > 0
        mock_extract.assert_called_once_with(str(pdf_path))

    @patch("knowledge_mcp_server.indexers.extract_text")
    def test_index_pdf_stores_documents(
        self, mock_extract, temp_store, mock_embeddings, empty_vectorstore, tmp_path
    ):
        """Test PDF indexing stores documents."""
        mock_extract.return_value = "Test content"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        ids = index_pdf(str(pdf_path), temp_store, mock_embeddings, empty_vectorstore)

        # Verify documents are stored
        for doc_id in ids:
            assert temp_store.get_doc(doc_id) is not None

    @patch("knowledge_mcp_server.indexers.extract_text")
    def test_index_pdf_adds_to_vectorstore(
        self, mock_extract, temp_store, mock_embeddings, empty_vectorstore, tmp_path
    ):
        """Test PDF indexing adds vectors to vectorstore."""
        mock_extract.return_value = "Test content for vectorstore"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        ids = index_pdf(str(pdf_path), temp_store, mock_embeddings, empty_vectorstore)

        assert len(empty_vectorstore.ids) == len(ids)
        assert len(empty_vectorstore.vectors) == len(ids)

    @patch("knowledge_mcp_server.indexers.extract_text")
    def test_index_pdf_metadata(
        self, mock_extract, temp_store, mock_embeddings, empty_vectorstore, tmp_path
    ):
        """Test PDF indexing includes correct metadata."""
        mock_extract.return_value = "Test"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        index_pdf(str(pdf_path), temp_store, mock_embeddings, empty_vectorstore)

        assert len(empty_vectorstore.metadatas) > 0
        assert "source" in empty_vectorstore.metadatas[0]
        assert "chunk_index" in empty_vectorstore.metadatas[0]


class TestIndexTex:
    """Test suite for index_tex function."""

    def test_index_tex_reads_file(
        self, temp_store, mock_embeddings, empty_vectorstore, tmp_path
    ):
        """Test TeX indexing reads file."""
        tex_path = tmp_path / "test.tex"
        tex_path.write_text("This is a test LaTeX document.")

        ids = index_tex(str(tex_path), temp_store, mock_embeddings, empty_vectorstore)

        assert len(ids) > 0

    def test_index_tex_strips_commands(
        self, temp_store, mock_embeddings, empty_vectorstore, tmp_path
    ):
        """Test TeX indexing strips LaTeX commands."""
        tex_content = r"This is \\textbf{bold} text and \\section{title}"
        tex_path = tmp_path / "test.tex"
        tex_path.write_text(tex_content)

        ids = index_tex(str(tex_path), temp_store, mock_embeddings, empty_vectorstore)

        # Verify stored text doesn't have LaTeX commands
        stored_text = temp_store.get_doc(ids[0])
        assert "\\textbf" not in stored_text
        assert "\\section" not in stored_text

    def test_index_tex_metadata(
        self, temp_store, mock_embeddings, empty_vectorstore, tmp_path
    ):
        """Test TeX indexing includes correct metadata."""
        tex_path = tmp_path / "test.tex"
        tex_path.write_text("Test content")

        index_tex(str(tex_path), temp_store, mock_embeddings, empty_vectorstore)

        assert "source" in empty_vectorstore.metadatas[0]
        assert str(tex_path) in empty_vectorstore.metadatas[0]["source"]


class TestIndexWebsite:
    """Test suite for index_website function."""

    @patch("knowledge_mcp_server.indexers.requests.get")
    def test_index_website_fetches_url(
        self, mock_get, temp_store, mock_embeddings, empty_vectorstore
    ):
        """Test website indexing fetches URL."""
        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        ids = index_website(
            "https://example.com", temp_store, mock_embeddings, empty_vectorstore
        )

        mock_get.assert_called_once_with("https://example.com", timeout=15)
        assert len(ids) > 0

    @patch("knowledge_mcp_server.indexers.requests.get")
    def test_index_website_strips_html(
        self, mock_get, temp_store, mock_embeddings, empty_vectorstore
    ):
        """Test website indexing strips HTML tags."""
        html = "<html><body><p>Content</p><script>alert('test')</script></body></html>"
        mock_response = Mock()
        mock_response.text = html
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        ids = index_website(
            "https://example.com", temp_store, mock_embeddings, empty_vectorstore
        )

        stored_text = temp_store.get_doc(ids[0])
        assert "<script>" not in stored_text
        assert "<p>" not in stored_text

    @patch("knowledge_mcp_server.indexers.requests.get")
    def test_index_website_metadata(
        self, mock_get, temp_store, mock_embeddings, empty_vectorstore
    ):
        """Test website indexing includes URL in metadata."""
        mock_response = Mock()
        mock_response.text = "<html><body>Test</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        url = "https://example.com/page"
        index_website(url, temp_store, mock_embeddings, empty_vectorstore)

        assert "source" in empty_vectorstore.metadatas[0]
        assert empty_vectorstore.metadatas[0]["source"] == url

    @patch("knowledge_mcp_server.indexers.requests.get")
    def test_index_website_error_handling(
        self, mock_get, temp_store, mock_embeddings, empty_vectorstore
    ):
        """Test website indexing handles errors."""
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            index_website(
                "https://example.com", temp_store, mock_embeddings, empty_vectorstore
            )
