import os
import uuid
import time
import configparser
from pathlib import Path

import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- Load config ---
config = configparser.ConfigParser()
config.read("config.ini")

EMBEDDING_MODEL      = config["EMBEDDING"]["model"]
EMBEDDING_DIM        = int(config["EMBEDDING"]["dim"])
EMBEDDING_BATCH_SIZE = int(config["EMBEDDING"]["batch_size"])
CHUNK_SIZE           = int(config["CHUNKING"]["chunk_size"])
CHUNK_OVERLAP        = int(config["CHUNKING"]["chunk_overlap"])
QDRANT_HOST          = config["QDRANT"]["host"]
QDRANT_PORT          = int(config["QDRANT"]["port"])

# Separate collection for local PDFs
PDF_COLLECTION_NAME  = "nih_definition"

# Directory containing your local PDF files
PDF_DIR = Path("/Users/carter/Desktop/NIH_RAG_Project/paper_files")

# --- Clients ---
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# --- Step 1: Create PDF collection ---
def create_collection(force_recreate: bool = False):
    existing = [c.name for c in qdrant.get_collections().collections]

    if PDF_COLLECTION_NAME in existing:
        if force_recreate:
            qdrant.delete_collection(collection_name=PDF_COLLECTION_NAME)
            print(f"Deleted existing collection: '{PDF_COLLECTION_NAME}'")
        else:
            print(f"Collection already exists: '{PDF_COLLECTION_NAME}' — appending")
            return

    qdrant.create_collection(
        collection_name=PDF_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    print(f"Created collection: '{PDF_COLLECTION_NAME}'")


# --- Step 2: Extract text + tables from PDF ---
def extract_pdf_content(pdf_path: Path) -> tuple[str, dict]:
    """
    Extract full text and metadata from a PDF using pdfplumber.
    Tables are converted to plain text rows and appended inline.
    Returns (full_text, metadata_dict).
    """
    full_text = []
    metadata  = {}

    with pdfplumber.open(pdf_path) as pdf:
        # Metadata from PDF properties
        meta = pdf.metadata or {}
        metadata = {
            "title":    meta.get("Title", pdf_path.stem),
            "author":   meta.get("Author", ""),
            "subject":  meta.get("Subject", ""),
            "filename": pdf_path.name,
            "pages":    len(pdf.pages),
        }

        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract regular text
            text = page.extract_text()
            if text:
                full_text.append(text.strip())

            # Extract tables — convert rows to readable text
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join(
                        cell.strip() if cell else ""
                        for cell in row
                    )
                    if row_text.strip():
                        full_text.append(f"[Table row, page {page_num}]: {row_text}")

    return "\n\n".join(full_text), metadata


# --- Step 3: Chunk text ---
def chunk_text(text: str) -> list[str]:
    """Word-level chunking with overlap — same settings as PMC pipeline."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# --- Step 4: Embed + upsert ---
def upsert_pdf(filename: str, chunks: list[str], metadata: dict):
    """Embed chunks and upsert into Qdrant with filename as the paper identifier."""
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
                "filename":     filename,           # acts as "folder" per paper
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "text":         chunk,
                "title":        metadata.get("title", ""),
                "author":       metadata.get("author", ""),
                "subject":      metadata.get("subject", ""),
                "pages":        metadata.get("pages", 0),
                "source":       "local_pdf",        # distinguishes from PMC pipeline
            }
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    qdrant.upsert(collection_name=PDF_COLLECTION_NAME, points=points)
    print(f"  Upserted {len(points)} chunks for '{filename}'")


# --- Main pipeline ---
def run_pdf_pipeline(force_recreate: bool = False):
    print("=" * 60)
    print("NIH PDF Ingestion Pipeline")
    print("=" * 60)

    # Validate PDF directory
    if not PDF_DIR.exists():
        print(f"ERROR: PDF directory not found: {PDF_DIR}")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")

    # Create collection
    create_collection(force_recreate=force_recreate)

    success, failed = [], []

    for i, pdf_path in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}] Processing '{pdf_path.name}'...")

        try:
            # Extract text + tables
            text, metadata = extract_pdf_content(pdf_path)

            if not text.strip():
                print(f"  No text extracted — skipping")
                failed.append(pdf_path.name)
                continue

            # Chunk
            chunks = chunk_text(text)
            print(f"  {len(chunks)} chunks from {len(text.split())} words | "
                  f"{metadata['pages']} pages | title: '{metadata['title'][:60]}'")

            # Embed + store
            upsert_pdf(pdf_path.name, chunks, metadata)
            success.append(pdf_path.name)

        except Exception as e:
            print(f"  ERROR processing '{pdf_path.name}': {e}")
            failed.append(pdf_path.name)

    # Summary
    print("\n" + "=" * 60)
    print(f"Pipeline complete.")
    print(f"  Success : {len(success)}")
    print(f"  Failed  : {len(failed)}")
    if failed:
        print(f"  Failed files: {failed}")
    count = qdrant.count(collection_name=PDF_COLLECTION_NAME)
    print(f"  Total vectors in '{PDF_COLLECTION_NAME}': {count.count}")
    print("=" * 60)


if __name__ == "__main__":
    run_pdf_pipeline(force_recreate=True)