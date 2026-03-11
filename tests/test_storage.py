"""Tests for storage module."""

import json

from knowledge_mcp_server.storage import LocalTextStore


class TestLocalTextStore:
    """Test suite for LocalTextStore class."""

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        root = tmp_path / "test_store"
        store = LocalTextStore(root=str(root))

        assert store.root == str(root)
        assert store.docs_dir.exists()
        assert store.meta_path.exists()

    def test_init_creates_empty_metadata(self, tmp_path):
        """Test that initialization creates empty metadata file."""
        root = tmp_path / "test_store"
        store = LocalTextStore(root=str(root))

        metadata = json.loads(store.meta_path.read_text())
        assert metadata == {}

    def test_add_doc_stores_text(self, temp_store):
        """Test adding a document stores the text."""
        temp_store.add_doc("doc1", "Hello, world!")

        doc_path = temp_store.docs_dir / "doc1.txt"
        assert doc_path.exists()
        assert doc_path.read_text(encoding="utf-8") == "Hello, world!"

    def test_add_doc_updates_metadata(self, temp_store):
        """Test adding a document updates metadata."""
        metadata = {"author": "Alice", "date": "2024-01-01"}
        temp_store.add_doc("doc1", "content", metadata=metadata)

        meta = temp_store._load_meta()
        assert "doc1" in meta
        assert meta["doc1"]["author"] == "Alice"
        assert meta["doc1"]["date"] == "2024-01-01"
        assert "path" in meta["doc1"]

    def test_add_doc_without_metadata(self, temp_store):
        """Test adding a document without metadata."""
        temp_store.add_doc("doc1", "content")

        meta = temp_store._load_meta()
        assert "doc1" in meta
        assert "path" in meta["doc1"]
        assert len(meta["doc1"]) == 1  # Only path

    def test_get_doc_retrieves_text(self, temp_store):
        """Test retrieving a document."""
        temp_store.add_doc("doc1", "Test content")

        result = temp_store.get_doc("doc1")
        assert result == "Test content"

    def test_get_doc_nonexistent_returns_none(self, temp_store):
        """Test retrieving nonexistent document returns None."""
        result = temp_store.get_doc("nonexistent")
        assert result is None

    def test_get_doc_missing_file_returns_none(self, temp_store):
        """Test retrieving document with missing file returns None."""
        temp_store.add_doc("doc1", "content")
        # Manually delete the file
        doc_path = temp_store.docs_dir / "doc1.txt"
        doc_path.unlink()

        result = temp_store.get_doc("doc1")
        assert result is None

    def test_list_docs_returns_all_metadata(self, temp_store):
        """Test listing all documents."""
        temp_store.add_doc("doc1", "content1", {"tag": "a"})
        temp_store.add_doc("doc2", "content2", {"tag": "b"})

        docs = temp_store.list_docs()
        assert len(docs) == 2
        assert "doc1" in docs
        assert "doc2" in docs
        assert docs["doc1"]["tag"] == "a"
        assert docs["doc2"]["tag"] == "b"

    def test_list_docs_empty_store(self, temp_store):
        """Test listing documents in empty store."""
        docs = temp_store.list_docs()
        assert docs == {}

    def test_add_doc_unicode_content(self, temp_store):
        """Test adding document with unicode content."""
        content = "Hello 世界 🌍"
        temp_store.add_doc("unicode_doc", content)

        result = temp_store.get_doc("unicode_doc")
        assert result == content
