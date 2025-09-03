import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running this file directly (from root or from scripts/)
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.docs_server import run_index_docs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Index docs/ into Chroma vector DB (db/).")
    parser.add_argument("--refresh", action="store_true", help="Clear and rebuild the vector DB")
    args = parser.parse_args(argv)
    msg = run_index_docs(refresh=args.refresh)
    print(msg)


if __name__ == "__main__":
    sys.exit(main())
