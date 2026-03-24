import requests
import time
import os
import uuid
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# NCBI
NCBI_SEARCH_URL         = config["NCBI"]["search_url"]
NCBI_SLEEP              = float(config["NCBI"]["sleep"])
SEARCH_LIMIT_PER_QUERY  = int(config["NCBI"]["search_limit_per_query"])

# AWS S3
PMC_S3_BASE_URL         = config["S3"]["base_url"]
S3_SLEEP                = float(config["S3"]["sleep"])

# Chunking
CHUNK_SIZE              = int(config["CHUNKING"]["chunk_size"])
CHUNK_OVERLAP           = int(config["CHUNKING"]["chunk_overlap"])

# Embedding
EMBEDDING_MODEL         = config["EMBEDDING"]["model"]
EMBEDDING_DIM           = int(config["EMBEDDING"]["dim"])
EMBEDDING_BATCH_SIZE    = int(config["EMBEDDING"]["batch_size"])

# Qdrant
QDRANT_HOST             = config["QDRANT"]["host"]
QDRANT_PORT             = int(config["QDRANT"]["port"])
COLLECTION_NAME         = "nih_stage_model"

# Queries
SEARCH_QUERIES = [
    config["QUERIES"]["primary"],
    config["QUERIES"]["case_study"],
]

load_dotenv()
EMAIL = os.getenv("NCBI_EMAIL")

# --- Clients ---
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# --- Step 1: Create collection (once) ---
def create_collection(force_recreate: bool = False):
    existing = [c.name for c in qdrant.get_collections().collections]
    
    if COLLECTION_NAME in existing:
        if force_recreate:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Deleted existing collection: '{COLLECTION_NAME}'")
        else:
            print(f"Collection already exists: '{COLLECTION_NAME}' — appending")
            return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    print(f"Created collection: '{COLLECTION_NAME}'")


# --- Step 2: Search PMC for a single query ---
def search_pmc(query: str, limit: int) -> list[str]:
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "email": EMAIL
    }
    r = requests.get(NCBI_SEARCH_URL, params=params)
    ids = r.json()["esearchresult"]["idlist"]
    pmcids = [f"PMC{id}" for id in ids]
    return pmcids


# --- Step 3: Run all queries and deduplicate ---
def collect_all_pmcids() -> list[str]:
    seen = set()
    all_pmcids = []

    for i, query in enumerate(SEARCH_QUERIES):
        print(f"\n[Query {i+1}/{len(SEARCH_QUERIES)}] Searching PMC...")
        pmcids = search_pmc(query, SEARCH_LIMIT_PER_QUERY)
        
        new = [p for p in pmcids if p not in seen]
        seen.update(new)
        all_pmcids.extend(new)
        
        print(f"  Retrieved: {len(pmcids)} | New after dedup: {len(new)} | Total so far: {len(all_pmcids)}")
        time.sleep(NCBI_SLEEP)

    print(f"\nTotal unique PMCIDs across all queries: {len(all_pmcids)}")
    return all_pmcids


# --- Step 4: Fetch full text from S3 over HTTPS (no local download) ---
def fetch_full_text(pmcid: str) -> str | None:
    for version in [1, 2, 3]:
        url = f"{PMC_S3_BASE_URL}/{pmcid}.{version}/{pmcid}.{version}.txt"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text
    print(f"  Could not fetch {pmcid} (tried versions 1-3)")
    return None


# --- Step 5: Fetch metadata from S3 JSON ---
def fetch_metadata(pmcid: str) -> dict:
    url = f"{PMC_S3_BASE_URL}/{pmcid}.1/{pmcid}.1.json"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        return r.json()
    return {}


# --- Step 6: Chunk text ---
def chunk_text(text: str) -> list[str]:
    """
    Word-level chunking with overlap.
    CHUNK_SIZE=400 words ≈ 500 tokens, fits bge-base-en-v1.5 context window.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 100:  # skip noise/tiny trailing chunks
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# --- Step 7: Embed + upsert to Qdrant ---
def upsert_paper(pmcid: str, chunks: list[str], metadata: dict):
    """
    Each chunk becomes one vector point.
    pmcid in payload acts as the logical 'folder' for the paper —
    use it to filter by paper at query time.
    """
    embeddings = embedder.encode(
        chunks,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "pmcid":        pmcid,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "text":         chunk,
                "title":        metadata.get("title", ""),
                "citation":     metadata.get("citation", ""),
                "license":      metadata.get("license_code", ""),
                "pmid":         str(metadata.get("pmid", "")),
                "doi":          metadata.get("doi", ""),
            }
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Upserted {len(points)} chunks for {pmcid}")


# --- Main pipeline ---
# force_recreate: bool = False -> Fresh start — wipes existing collection
#                      = True  -> Append mode — keeps existing vectors
def run_pipeline(force_recreate: bool = True):
    print("=" * 60)
    print("NIH RAG Ingestion Pipeline")
    print("=" * 60)

    # Create Qdrant collection if needed
    create_collection(force_recreate=force_recreate)

    # Collect all unique PMCIDs across all queries
    pmcids = collect_all_pmcids()

    success, failed = [], []

    for i, pmcid in enumerate(pmcids):
        print(f"\n[{i+1}/{len(pmcids)}] Processing {pmcid}...")

        # Fetch full text (streaming, no local file)
        text = fetch_full_text(pmcid)
        if not text:
            failed.append(pmcid)
            continue

        # Fetch metadata (title, citation, doi, etc.)
        metadata = fetch_metadata(pmcid)

        # Chunk
        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks from {len(text.split())} words")

        # Embed + store in Qdrant
        upsert_paper(pmcid, chunks, metadata)
        success.append(pmcid)

        time.sleep(S3_SLEEP)

    # Final summary
    print("\n" + "=" * 60)
    print(f"Pipeline complete.")
    print(f"  Success : {len(success)}")
    print(f"  Failed  : {len(failed)}")
    if failed:
        print(f"  Failed PMCIDs: {failed}")
    count = qdrant.count(collection_name=COLLECTION_NAME)
    print(f"  Total vectors in '{COLLECTION_NAME}': {count.count}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()