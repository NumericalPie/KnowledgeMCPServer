from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

from .embeddings import Embeddings
from .indexers import index_pdf, index_tex, index_website
from .storage import LocalTextStore
from .vectorstore import InMemoryVectorStore


def _load_or_create_vecstore(root: str) -> InMemoryVectorStore:
    try:
        vecstore = InMemoryVectorStore.load(root=root)
        logger.info("Loaded existing vector store")
    except (FileNotFoundError, OSError):
        vecstore = InMemoryVectorStore()
        logger.info("Created new vector store")
    except Exception as exc:  # pragma: no cover - defensive fallback
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        logger.warning("Failed to load existing vector store: {}", exc)
        vecstore = InMemoryVectorStore()
        logger.info("Created new vector store")

    return vecstore


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser("knowledge_mcp_server CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index")
    p_index.add_argument("--pdf", help="Path to PDF to index")
    p_index.add_argument("--tex", help="Path to TeX file to index")
    p_index.add_argument("--url", help="URL to index")

    args = parser.parse_args(argv)

    store = LocalTextStore()
    embeddings = Embeddings()
    # Load existing vector store or create new one
    vecstore = _load_or_create_vecstore(store.root)

    if args.cmd == "index":
        indexed = False
        if args.pdf:
            ids = index_pdf(args.pdf, store, embeddings, vecstore)
            logger.info("Indexed PDF chunks: {}", len(ids))
            indexed = True
        if args.tex:
            ids = index_tex(args.tex, store, embeddings, vecstore)
            logger.info("Indexed TeX chunks: {}", len(ids))
            indexed = True
        if args.url:
            ids = index_website(args.url, store, embeddings, vecstore)
            logger.info("Indexed URL chunks: {}", len(ids))
            indexed = True
        if not (args.pdf or args.tex or args.url):
            logger.warning("No input specified. Use --pdf, --tex or --url")
        if indexed:
            try:
                vecstore.save(root=store.root)
                logger.info("Vector store saved successfully")
            except OSError as e:
                logger.warning("Failed to save vector store: {}", e)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
