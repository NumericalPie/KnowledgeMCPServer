"""Tests for CLI module."""

from unittest.mock import MagicMock, patch

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
    def test_index_pdf_command(
        self, _mock_emb, _mock_store, mock_vec, mock_index, capsys
    ):
        """Test indexing PDF via CLI."""
        mock_vec_instance = MagicMock()
        mock_vec.load.return_value = mock_vec_instance
        mock_index.return_value = ["id1", "id2", "id3"]

        main(["index", "--pdf", "test.pdf"])

        mock_index.assert_called_once()
        captured = capsys.readouterr()
        assert "3" in captured.out

    @patch("knowledge_mcp_server.cli.index_tex")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_index_tex_command(
        self, _mock_emb, _mock_store, mock_vec, mock_index, capsys
    ):
        """Test indexing TeX via CLI."""
        mock_vec_instance = MagicMock()
        mock_vec.load.return_value = mock_vec_instance
        mock_index.return_value = ["id1"]

        main(["index", "--tex", "test.tex"])

        mock_index.assert_called_once()
        captured = capsys.readouterr()
        assert "1" in captured.out

    @patch("knowledge_mcp_server.cli.index_website")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_index_url_command(
        self, _mock_emb, _mock_store, mock_vec, mock_index, capsys
    ):
        """Test indexing URL via CLI."""
        mock_vec_instance = MagicMock()
        mock_vec.load.return_value = mock_vec_instance
        mock_index.return_value = ["id1", "id2"]

        main(["index", "--url", "https://example.com"])

        mock_index.assert_called_once()
        captured = capsys.readouterr()
        assert "2" in captured.out

    @patch("knowledge_mcp_server.cli.index_website")
    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_loads_existing_vectorstore(self, _mock_emb, _mock_store, mock_vec, mock_index):
        """Test CLI loads existing vector store on startup."""
        mock_vec_instance = MagicMock()
        mock_vec.load.return_value = mock_vec_instance
        mock_index.return_value = ["id1"]

        main(["index", "--url", "https://example.com"])

        mock_vec.load.assert_called_once()

    @patch("knowledge_mcp_server.cli.InMemoryVectorStore")
    @patch("knowledge_mcp_server.cli.LocalTextStore")
    @patch("knowledge_mcp_server.cli.Embeddings")
    def test_creates_new_vectorstore_on_error(
        self, _mock_emb, _mock_store, mock_vec, capsys
    ):
        """Test CLI creates new vector store if load fails."""
        mock_vec.load.side_effect = Exception("Load failed")

        main([])

        mock_vec.assert_called()
        captured = capsys.readouterr()
        assert "Created new vector store" in captured.out

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
    def test_no_input_specified_message(self, _mock_emb, _mock_store, mock_vec, capsys):
        """Test message when no input is specified."""
        mock_vec.load.return_value = MagicMock()

        main(["index"])

        captured = capsys.readouterr()
        assert "No input specified" in captured.out
