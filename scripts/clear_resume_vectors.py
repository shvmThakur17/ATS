"""Safely clear resume vectors from the configured Pinecone index.

Preview by default. Deletes only namespaces matching:
    <configured namespace prefix>:<64-character lowercase SHA-256>

The Pinecone index itself is never deleted.
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from app.parsing.vector_store.models import VectorStoreConfig
from app.parsing.vector_store.pinecone_store import PineconeStore

DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")

def main() -> int:
    parser = argparse.ArgumentParser(description="Preview or delete resume namespaces from Pinecone.")
    parser.add_argument("--delete",action="store_true",help="Actually delete matching namespaces. Without this flag, preview only.",)
    args = parser.parse_args()
    load_dotenv()
    try:
        config = VectorStoreConfig.from_env()
        store = PineconeStore(config)
        if not store.index_exists():
            print(f"Configured index does not exist: {config.index_name}")
            return 1

        stats = store.index.describe_index_stats()
    except Exception as exc:
        print(f"Could not inspect Pinecone: {type(exc).__name__}: {exc}")
        return 1

    prefix = f"{config.namespace_prefix}:"
    targets = []

    for namespace, details in stats.namespaces.items():
        if not namespace.startswith(prefix):
            continue

        document_id = namespace[len(prefix):]
        if DOCUMENT_ID_PATTERN.fullmatch(document_id):
            targets.append((namespace, int(details.vector_count)))

    targets.sort()

    print(f"Index: {config.index_name}")
    print(f"Resume namespace prefix: {prefix}")
    print(f"Matching namespaces: {len(targets)}")
    print(f"Vectors in matching namespaces: {sum(count for _, count in targets)}")

    for namespace, count in targets:
        print(f"  {namespace}  vectors={count}")

    if not targets:
        print("No matching resume namespaces found. Nothing to do.")
        return 0

    if not args.delete:
        print("\nPREVIEW ONLY. No data was deleted.")
        print("Review the targets, then run with --delete to proceed.")
        return 0

    expected_confirmation = f"DELETE {config.index_name} {len(targets)}"
    print("\nWARNING: This permanently deletes all vectors in the listed namespaces.")
    confirmation = input(f"Type exactly: {expected_confirmation}\n> ").strip()

    if confirmation != expected_confirmation:
        print("Confirmation did not match. No data was deleted.")
        return 2

    deleted = 0
    failed = []

    for namespace, count in targets:
        try:
            store.delete_namespace(namespace)
            deleted += 1
            print(f"Deletion requested: {namespace} ({count} vectors)")
        except Exception as exc:
            failed.append(namespace)
            print(f"FAILED: {namespace} ({type(exc).__name__})")

    print(f"\nDeletion requests completed: {deleted}/{len(targets)}")

    try:
        after = store.index.describe_index_stats()
        remaining = [
            (namespace, int(details.vector_count))
            for namespace, details in after.namespaces.items()
            if namespace.startswith(prefix)
            and DOCUMENT_ID_PATTERN.fullmatch(namespace[len(prefix):])
            and int(details.vector_count) > 0
        ]
        if remaining:
            print("Some vectors may remain while Pinecone processes deletions:")
            for namespace, count in remaining:
                print(f"  {namespace}: {count} vectors")
        elif not failed:
            print("Verification: no matching resume vectors remain visible.")
    except Exception as exc:
        print(f"Verification unavailable: {type(exc).__name__}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
