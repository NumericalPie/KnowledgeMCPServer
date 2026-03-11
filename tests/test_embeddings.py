"""Tests for embeddings module."""

from unittest.mock import MagicMock, patch

import numpy as np

from knowledge_mcp_server.embeddings import Embeddings


class TestEmbeddings:
    """Test suite for Embeddings class."""

    @patch("knowledge_mcp_server.embeddings.SentenceTransformer")
    def test_init_default_model(self, mock_transformer):
        """Test initialization with default model."""
        emb = Embeddings()
        assert emb.model_name == "all-MiniLM-L6-v2"
        mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("knowledge_mcp_server.embeddings.SentenceTransformer")
    def test_init_custom_model(self, mock_transformer):
        """Test initialization with custom model."""
        emb = Embeddings(model_name="custom-model")
        assert emb.model_name == "custom-model"
        mock_transformer.assert_called_once_with("custom-model")

    @patch("knowledge_mcp_server.embeddings.SentenceTransformer")
    def test_embed_returns_numpy_array(self, mock_transformer):
        """Test that embed returns numpy array."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_transformer.return_value = mock_model

        emb = Embeddings()
        texts = ["hello", "world"]
        result = emb.embed(texts)

        mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    @patch("knowledge_mcp_server.embeddings.SentenceTransformer")
    def test_embed_empty_list(self, mock_transformer):
        """Test embedding empty list."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([])
        mock_transformer.return_value = mock_model

        emb = Embeddings()
        result = emb.embed([])

        mock_model.encode.assert_called_once_with([], convert_to_numpy=True)
        assert isinstance(result, np.ndarray)
