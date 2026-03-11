"""Shared pytest fixtures for tests."""

import numpy as np
import pytest

from knowledge_mcp_server.storage import LocalTextStore
from knowledge_mcp_server.vectorstore import InMemoryVectorStore


@pytest.fixture
def mock_embeddings():
    """Mock embeddings that return deterministic vectors."""

    class MockEmbeddings:
        def __init__(self, model_name: str = "mock-model"):
            self.model_name = model_name

        def embed(self, texts: list[str]):
            # Return simple deterministic vectors based on text length
            return np.array([[float(len(t)), 1.0, 0.5] for t in texts], dtype=np.float32)

    return MockEmbeddings()


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary LocalTextStore."""
    return LocalTextStore(root=str(tmp_path / "data"))


@pytest.fixture
def empty_vectorstore():
    """Create an empty InMemoryVectorStore."""
    return InMemoryVectorStore()


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing."""
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
        dtype=np.float32,
    )


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return [
        {"source": "doc1.pdf", "chunk_index": 0},
        {"source": "doc1.pdf", "chunk_index": 1},
        {"source": "doc2.pdf", "chunk_index": 0},
        {"source": "doc2.pdf", "chunk_index": 1},
    ]
