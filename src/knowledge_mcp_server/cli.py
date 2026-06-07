import argparse
import logging
import sys
from collections.abc import Sequence

from .embeddings import Embeddings
from .indexers import index_pdf, index_tex, index_website
from .storage import LocalTextStore
from .vectorstore import InMemoryVectorStore

logger = logging.getLogger(__name__)


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
        logger.warning("Failed to load existing vector store: %s", exc)
        vecstore = InMemoryVectorStore()
        logger.info("Created new vector store")

    return vecstore


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
            logger.info("Indexed PDF chunks: %d", len(ids))
            indexed = True
        if args.tex:
            ids = index_tex(args.tex, store, embeddings, vecstore)
            logger.info("Indexed TeX chunks: %d", len(ids))
            indexed = True
        if args.url:
            ids = index_website(args.url, store, embeddings, vecstore)
            logger.info("Indexed URL chunks: %d", len(ids))
            indexed = True
        if not (args.pdf or args.tex or args.url):
            logger.warning("No input specified. Use --pdf, --tex or --url")
        if indexed:
            try:
                vecstore.save(root=store.root)
                logger.info("Vector store saved successfully")
            except OSError as e:
                logger.warning("Failed to save vector store: %s", e)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
