"""Tests for vectorstore module."""

import json

import numpy as np
import pytest

from knowledge_mcp_server.vectorstore import InMemoryVectorStore


class TestInMemoryVectorStore:
    """Test suite for InMemoryVectorStore class."""

    def test_init_creates_empty_store(self, empty_vectorstore):
        """Test initialization creates empty store."""
        assert len(empty_vectorstore.ids) == 0
        assert len(empty_vectorstore.vectors) == 0
        assert len(empty_vectorstore.metadatas) == 0

    def test_add_vectors(self, empty_vectorstore, sample_vectors, sample_metadata):
        """Test adding vectors to store."""
        ids = ["id1", "id2", "id3", "id4"]
        empty_vectorstore.add(ids, list(sample_vectors), sample_metadata)

        assert len(empty_vectorstore.ids) == 4
        assert len(empty_vectorstore.vectors) == 4
        assert len(empty_vectorstore.metadatas) == 4
        assert empty_vectorstore.ids == ids

    def test_add_vectors_without_metadata(self, empty_vectorstore, sample_vectors):
        """Test adding vectors without metadata."""
        ids = ["id1", "id2"]
        vectors = list(sample_vectors[:2])
        empty_vectorstore.add(ids, vectors)

        assert len(empty_vectorstore.metadatas) == 2
        assert empty_vectorstore.metadatas == [{}, {}]

    def test_search_returns_top_k(self, empty_vectorstore, sample_vectors, sample_metadata):
        """Test search returns correct number of results."""
        ids = ["id1", "id2", "id3", "id4"]
        empty_vectorstore.add(ids, list(sample_vectors), sample_metadata)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = empty_vectorstore.search(query, top_k=2)

        assert len(results) == 2

    def test_search_returns_highest_similarity(
        self, empty_vectorstore, sample_vectors, sample_metadata
    ):
        """Test search returns results ordered by similarity."""
        ids = ["id1", "id2", "id3", "id4"]
        empty_vectorstore.add(ids, list(sample_vectors), sample_metadata)

        # Query identical to first vector
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = empty_vectorstore.search(query, top_k=3)

        # First result should be id1 with highest score
        assert results[0][0] == "id1"
        assert results[0][1] > results[1][1]  # Higher similarity

    def test_search_empty_store(self, empty_vectorstore):
        """Test search on empty store returns empty list."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = empty_vectorstore.search(query, top_k=5)

        assert results == []

    def test_search_returns_metadata(
        self, empty_vectorstore, sample_vectors, sample_metadata
    ):
        """Test search results include metadata."""
        ids = ["id1", "id2"]
        empty_vectorstore.add(ids, list(sample_vectors[:2]), sample_metadata[:2])

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = empty_vectorstore.search(query, top_k=2)

        assert results[0][2] == sample_metadata[0]

    def test_save_creates_files(self, empty_vectorstore, sample_vectors, tmp_path):
        """Test save creates necessary files."""
        ids = ["id1", "id2"]
        empty_vectorstore.add(ids, list(sample_vectors[:2]))
        empty_vectorstore.save(root=str(tmp_path))

        vectorstore_dir = tmp_path / "vectorstore"
        assert vectorstore_dir.exists()
        assert (vectorstore_dir / "ids.json").exists()
        assert (vectorstore_dir / "metadatas.json").exists()
        assert (vectorstore_dir / "vectors.npy").exists()

    def test_save_persists_data(self, empty_vectorstore, sample_vectors, tmp_path):
        """Test save persists correct data."""
        ids = ["id1", "id2"]
        metadata = [{"a": 1}, {"b": 2}]
        empty_vectorstore.add(ids, list(sample_vectors[:2]), metadata)
        empty_vectorstore.save(root=str(tmp_path))

        vectorstore_dir = tmp_path / "vectorstore"
        saved_ids = json.loads((vectorstore_dir / "ids.json").read_text())
        saved_meta = json.loads((vectorstore_dir / "metadatas.json").read_text())

        assert saved_ids == ids
        assert saved_meta == metadata

    def test_load_restores_data(self, empty_vectorstore, sample_vectors, tmp_path):
        """Test load restores saved data."""
        ids = ["id1", "id2"]
        metadata = [{"a": 1}, {"b": 2}]
        empty_vectorstore.add(ids, list(sample_vectors[:2]), metadata)
        empty_vectorstore.save(root=str(tmp_path))

        loaded = InMemoryVectorStore.load(root=str(tmp_path))

        assert loaded.ids == ids
        assert loaded.metadatas == metadata
        assert len(loaded.vectors) == 2

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """Test load from nonexistent directory returns empty store."""
        loaded = InMemoryVectorStore.load(root=str(tmp_path / "nonexistent"))

        assert len(loaded.ids) == 0
        assert len(loaded.vectors) == 0
        assert len(loaded.metadatas) == 0

    def test_load_corrupted_data_returns_empty(self, tmp_path):
        """Test load with corrupted data returns empty store."""
        vectorstore_dir = tmp_path / "vectorstore"
        vectorstore_dir.mkdir(parents=True)
        (vectorstore_dir / "ids.json").write_text("invalid json")

        loaded = InMemoryVectorStore.load(root=str(tmp_path))

        assert len(loaded.ids) == 0

    def test_save_empty_store(self, empty_vectorstore, tmp_path):
        """Test saving empty store."""
        empty_vectorstore.save(root=str(tmp_path))

        vectorstore_dir = tmp_path / "vectorstore"
        assert vectorstore_dir.exists()

        saved_ids = json.loads((vectorstore_dir / "ids.json").read_text())
        assert saved_ids == []

    def test_cosine_similarity_calculation(self, empty_vectorstore):
        """Test cosine similarity is calculated correctly."""
        # Orthogonal vectors should have 0 similarity
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)

        sim = empty_vectorstore._cosine_sim(v1, np.array([v2]))
        assert pytest.approx(sim[0], abs=1e-6) == 0.0

        # Identical vectors should have 1.0 similarity
        sim = empty_vectorstore._cosine_sim(v1, np.array([v1]))
        assert pytest.approx(sim[0], abs=1e-6) == 1.0
