import json
from pathlib import Path

import numpy as np


class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors: list[np.ndarray] = []
        self.metadatas: list[dict] = []

    def add(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadatas: list[dict] | None = None,
    ) -> None:
        self.ids.extend(ids)
        self.vectors.extend(list(vectors))
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(ids))

    def save(self, root: str = "data") -> None:
        """Persist ids, vectors and metadatas under `root/vectorstore/`.

        - vectors.npy : numpy array of shape (N, dim)
        - ids.json : list of ids
        - metadatas.json : list of metadata dicts
        """

        root_path = Path(root)
        root_path.mkdir(exist_ok=True)
        folder = root_path / "vectorstore"
        folder.mkdir(exist_ok=True)
        ids_path = folder / "ids.json"
        meta_path = folder / "metadatas.json"
        vec_path = folder / "vectors.npy"
        # save ids and metadatas
        ids_path.write_text(json.dumps(self.ids, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(
            json.dumps(self.metadatas, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # save vectors as numpy array
        if self.vectors:
            arr = np.vstack(self.vectors)
            np.save(vec_path, arr)
        else:
            np.save(vec_path, np.zeros((0,)))

    @classmethod
    def load(cls, root: str = "data") -> "InMemoryVectorStore":
        """Load a persisted vectorstore from disk into memory. If not found, returns empty store."""

        folder = Path(root) / "vectorstore"
        inst = cls()
        ids_path = folder / "ids.json"
        meta_path = folder / "metadatas.json"
        vec_path = folder / "vectors.npy"
        if not folder.exists():
            return inst
        try:
            inst.ids = json.loads(ids_path.read_text(encoding="utf-8"))
            inst.metadatas = json.loads(meta_path.read_text(encoding="utf-8"))
            arr = np.load(vec_path, allow_pickle=False)
            # ensure 2D
            if arr.size == 0:
                inst.vectors = []
            else:
                inst.vectors = [arr[i] for i in range(arr.shape[0])]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # on any error, return empty store
            return cls()
        return inst

    def _cosine_sim(self, q: np.ndarray, vs: np.ndarray) -> np.ndarray:
        q_norm = q / np.linalg.norm(q)
        vs_norm = vs / np.linalg.norm(vs, axis=1, keepdims=True)
        return vs_norm.dot(q_norm)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict]]:
        if not self.vectors:
            return []
        vs = np.vstack(self.vectors)
        sims = self._cosine_sim(query_vector, vs)
        idx = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i]), self.metadatas[i]) for i in idx]
