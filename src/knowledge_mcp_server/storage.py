import json
from pathlib import Path


class LocalTextStore:
    """Simple local text storage for documents and metadata.

    Stores each document as a text file under `root/docs/<doc_id>.txt` and writes
    metadata to `root/metadata.json`.
    """

    def __init__(self, root: str = "data") -> None:
        self.root = root
        self.root_path = Path(root)
        self.docs_dir = self.root_path / "docs"
        self.meta_path = self.root_path / "metadata.json"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            self.meta_path.write_text(json.dumps({}), encoding="utf-8")

    def _load_meta(self) -> dict[str, dict]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, meta: dict[str, dict]) -> None:
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_doc(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        path = self.docs_dir / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8")
        meta = self._load_meta()
        meta[doc_id] = metadata or {}
        meta[doc_id]["path"] = str(path)
        self._write_meta(meta)

    def get_doc(self, doc_id: str) -> str | None:
        meta = self._load_meta()
        entry = meta.get(doc_id)
        if not entry:
            return None
        path_str = entry.get("path")
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_docs(self) -> dict[str, dict]:
        return self._load_meta()
