"""Tests for CLI module."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from knowledge_mcp_server.cli import main


class TestCLI:
    """Test suite for CLI."""

    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_main_no_args_shows_help(self, _mock_emb, _mock_store, mock_vec, capsys):
        """Test running without arguments shows help."""
        mock_vec.load.return_value = MagicMock()

        main([])

        captured = capsys.readouterr()
        assert "usage:" in captured.out or "usage:" in captured.err

    @patch("knowledge_mcp_server.cli.index_pdf")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_index_pdf_command(self, _mock_emb, _mock_store, mock_vec, mock_index, caplog):
        """Test indexing PDF via CLI."""
        mock_vec.load.return_value = MagicMock()
        mock_index.return_value = ["id1", "id2", "id3"]

        with caplog.at_level(logging.INFO, logger="knowledge_mcp_server.cli"):
            main(["index", "--pdf", "test.pdf"])

        mock_index.assert_called_once()
        assert "3" in caplog.text

    @patch("knowledge_mcp_server.cli.index_tex")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_index_tex_command(self, _mock_emb, _mock_store, mock_vec, mock_index, caplog):
        """Test indexing TeX via CLI."""
        mock_vec.load.return_value = MagicMock()
        mock_index.return_value = ["id1"]

        with caplog.at_level(logging.INFO, logger="knowledge_mcp_server.cli"):
            main(["index", "--tex", "test.tex"])

        mock_index.assert_called_once()
        assert "1" in caplog.text

    @patch("knowledge_mcp_server.cli.index_website")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_index_url_command(self, _mock_emb, _mock_store, mock_vec, mock_index, caplog):
        """Test indexing URL via CLI."""
        mock_vec.load.return_value = MagicMock()
        mock_index.return_value = ["id1", "id2"]

        with caplog.at_level(logging.INFO, logger="knowledge_mcp_server.cli"):
            main(["index", "--url", "https://example.com"])

        mock_index.assert_called_once()
        assert "2" in caplog.text

    @patch("knowledge_mcp_server.cli.index_website")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_loads_existing_vectorstore(self, _mock_emb, _mock_store, mock_vec, mock_index):
        """Test CLI loads existing vector store on startup."""
        mock_vec.load.return_value = MagicMock()
        mock_index.return_value = ["id1"]

        main(["index", "--url", "https://example.com"])

        mock_vec.load.assert_called_once()

    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_creates_new_vectorstore_on_error(
        self, _mock_emb, _mock_store, mock_vec, caplog
    ):
        """Test CLI creates new vector store if load fails."""
        mock_vec.load.side_effect = Exception("Load failed")

        with caplog.at_level(logging.INFO, logger="knowledge_mcp_server.cli"):
            main([])

        mock_vec.assert_called()
        assert "Created new vector store" in caplog.text

    @patch("knowledge_mcp_server.cli.index_pdf")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_saves_vectorstore_after_indexing(
        self, _mock_emb, _mock_store, mock_vec, mock_index
    ):
        """Test CLI saves vector store after successful indexing."""
        mock_vec_instance = MagicMock()
        mock_vec.load.return_value = mock_vec_instance
        mock_index.return_value = ["id1"]

        main(["index", "--pdf", "test.pdf"])

        mock_vec_instance.save.assert_called_once()

    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_no_input_specified_message(self, _mock_emb, _mock_store, mock_vec, caplog):
        """Test message when no input is specified."""
        mock_vec.load.return_value = MagicMock()

        with caplog.at_level(logging.WARNING, logger="knowledge_mcp_server.cli"):
            main(["index"])

        assert "No input specified" in caplog.text

    @pytest.mark.parametrize(
        ("argv", "expected_log"),
        [
            (["index", "--pdf", "a.pdf"], "PDF chunks"),
            (["index", "--tex", "a.tex"], "TeX chunks"),
            (["index", "--url", "https://x.com"], "URL chunks"),
        ],
    )
    @patch("knowledge_mcp_server.cli.index_website")
    @patch("knowledge_mcp_server.cli.index_tex")
    @patch("knowledge_mcp_server.cli.index_pdf")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_index_logs_chunk_type(
        self,
        _mock_emb,
        _mock_store,
        mock_vec,
        mock_pdf,
        mock_tex,
        mock_url,
        argv,
        expected_log,
        caplog,
    ):
        """Test that each index subcommand logs the correct chunk type."""
        mock_vec.load.return_value = MagicMock()
        mock_pdf.return_value = ["id1"]
        mock_tex.return_value = ["id1"]
        mock_url.return_value = ["id1"]

        with caplog.at_level(logging.INFO, logger="knowledge_mcp_server.cli"):
            main(argv)

        assert expected_log in caplog.text
