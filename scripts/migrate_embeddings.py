"""
Run this ONCE after upgrading to BAAI/bge-m3.
Re-embeds all documents in Supabase with the new 1024-dim model.
Usage: python migrate_embeddings.py [--dry-run]
"""

import argparse
import os
import sys
from typing import List

from tqdm import tqdm

try:
    from supabase import create_client
except Exception as exc:
    print('supabase package not available:', exc)
    sys.exit(1)

from embeddings_rag import HuggingFaceEmbeddings

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
TABLE = os.environ.get('SUPABASE_TABLE', 'documents')
MODEL_NAME = os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3')
BATCH_SIZE = 50

if not SUPABASE_URL or not SUPABASE_KEY:
    print('Please set SUPABASE_URL and SUPABASE_KEY environment variables.')
    sys.exit(1)

client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_docs() -> List[dict]:
    resp = client.table(TABLE).select('id, content, metadata').execute()
    return resp.data or []


def upsert_batch(rows: List[dict]):
    return client.table(TABLE).upsert(rows).execute()


def main() -> None:
    parser = argparse.ArgumentParser(description='Migrate Supabase document embeddings to BAAI/bge-m3 vector(1024).')
    parser.add_argument('--dry-run', action='store_true', help='Count documents without modifying anything.')
    args = parser.parse_args()

    docs = fetch_all_docs()
    total = len(docs)
    print(f'Found {total} chunks to re-embed.')

    alter_sql = f'ALTER TABLE {TABLE} ALTER COLUMN embedding TYPE vector({1024});'
    print('Before running this script, ensure the embedding column is resized in Supabase:')
    print(alter_sql)

    if args.dry_run:
        print('Dry run mode enabled. No changes will be written.')

    embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    succeeded = 0
    failed = 0

    for start in range(0, total, BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        batch_ids = [row.get('id') for row in batch]
        try:
            texts = [row.get('content') or '' for row in batch]
            embeddings = embedder.embed_documents(texts)

            if not args.dry_run:
                rows = []
                for row, emb in zip(batch, embeddings):
                    rows.append({
                        'id': row.get('id'),
                        'content': row.get('content') or '',
                        'metadata': row.get('metadata') or {},
                        'embedding': emb,
                    })
                upsert_batch(rows)

            succeeded += len(batch)
            print(f'Migrated {min(start + BATCH_SIZE, total)}/{total} chunks...')
        except Exception as exc:
            failed += len(batch)
            print(f'Error migrating batch {start + 1}-{min(start + BATCH_SIZE, total)}: {exc}')
            continue

    print(f'Migration complete. {succeeded} succeeded, {failed} failed.')


if __name__ == '__main__':
    main()
