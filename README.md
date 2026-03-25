# NIH Stage Model RAG Chatbot

A research chatbot built to support behavioral intervention researchers working within the **NIH Stage Model** framework. The system uses Retrieval-Augmented Generation (RAG) to ground responses in peer-reviewed literature, surfacing relevant case studies, methodologies, and stage-specific guidance based on where a researcher is in their intervention development journey.

---

## Project Overview

The NIH Stage Model (Rounsaville et al.) provides a structured framework for developing and testing behavioral interventions across five stages — from basic science and manual development through efficacy trials, effectiveness research, and ultimately large-scale dissemination. Researchers new to the framework often struggle to find relevant prior work at their specific stage.

This chatbot addresses that gap by:

- Ingesting open-access PMC literature related to the NIH Stage Model
- Ingesting local PDF case studies and methodology papers
- Storing all content as searchable vector embeddings in Qdrant
- Allowing researchers to ask natural language questions and receive grounded, cited responses from Qwen2.5:3b running locally via Ollama

---

## Architecture

```
PMC eSearch API ──→ PMC S3 (HTTPS) ──→ Text Extraction
                                              │
Local PDF Files ──→ pdfplumber  ──────────────┤
                                              ▼
                                    Chunking (400 words, 50 overlap)
                                              │
                                              ▼
                                 BAAI/bge-base-en-v1.5 Embedder
                                              │
                                              ▼
                                    Qdrant Vector Database
                                     ┌────────────────┐
                                     │ nih_stage_model│  ← PMC papers
                                     │ nih_stage_model│
                                     │     _pdfs      │  ← Local PDFs
                                     └────────────────┘
                                              │
                                    User Query (embedded)
                                              │
                                     Top-K Retrieval
                                              │
                                     RAG Prompt Builder
                                              │
                                    Qwen2.5:3b (Ollama)
                                              │
                                       Response + Sources
```

---

## RAG Infrastructure

### Vector Database — Qdrant (Docker)

Qdrant runs locally via Docker. Each collection stores vector embeddings alongside rich metadata payloads, allowing both semantic search and structured filtering by paper, stage, or source type.

```bash
docker run -d --name my_qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Dashboard available at `http://localhost:6333/dashboard`.

### Collections

| Collection | Source | Contents |
|---|---|---|
| `nih_stage_model` | PMC open-access S3 | Papers retrieved via eSearch queries |
| `nih_stage_model_pdfs` | Local `/paper_files/` directory | Manually curated PDF case studies |

Each vector point payload includes:

```json
{
  "pmcid":        "PMC12345678",
  "chunk_index":  3,
  "total_chunks": 47,
  "text":         "The Stage I pilot involved...",
  "title":        "A Stage Model approach to...",
  "citation":     "J Consult Clin Psychol. 2023...",
  "license":      "CC BY",
  "pmid":         "12345678",
  "doi":          "10.1000/xyz123"
}
```

The `pmcid` or `filename` field acts as a logical grouping key — equivalent to a per-paper folder — and can be used to filter results to a single paper at query time.

### Embedding Model

**BAAI/bge-base-en-v1.5** (768 dimensions, cosine similarity)

Chosen for its strong performance on academic and scientific text, lightweight footprint (runs on CPU), and compatibility with the retrieval quality ceiling of Qwen2.5:3b.

### Language Model

**Qwen2.5:3b-instruct** served locally via Ollama at `http://localhost:11434`.

---

## Building the Database

### Method 1 — PMC eSearch + S3 Pipeline (Primary)

Queries the NCBI PMC database via eSearch, collects PMCIDs, then fetches full-text `.txt` files directly from PMC's public S3 bucket over HTTPS — no local file storage required.

```
eSearch → PMCIDs → S3 HTTPS fetch → chunk → embed → Qdrant
```

Two complementary search queries are run and deduplicated:

**Primary query** — broad NIH Stage Model terminology:
```
"NIH Stage Model"[tw] OR
"stage model of behavioral intervention"[tw] OR
"Rounsaville"[tw] OR
"behavioral intervention development stages"[tw] OR
"Stage 0 intervention"[tw] OR
"Stage 1 intervention development"[tw]
```

**Case study query** — stage-specific empirical work:
```
("pilot feasibility"[tw] OR "Stage I feasibility"[tw] OR
"Stage II efficacy trial"[tw] OR "Stage III effectiveness"[tw] OR
"intervention development framework"[tw])
AND ("NIH Stage"[tw] OR "behavioral intervention"[tw])
```

Both queries are filtered to `open_access[filter] OR author_manuscript[filter]` for full-text access.

Run the PMC pipeline:
```bash
python nih_rag_pipeline.py
```

### Method 2 — Local PDF Ingestion

Processes manually curated PDF files from `/paper_files/` using `pdfplumber`, which preserves table content as readable inline text rows alongside body text.

```
PDF files → pdfplumber → text + tables → chunk → embed → Qdrant
```

Run the PDF pipeline:
```bash
python pdf_ingest_pipeline.py
```

### Method 3 — BioC API (Alternative to S3)

PMC's BioC API returns structured JSON with passage-level section labels (`ABSTRACT`, `INTRO`, `METHODS`, `RESULTS`). This allows section-aware chunking — useful if you want to weight abstract or methods chunks differently at retrieval time.

```python
url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
```

Each passage in the response includes `infons.section_type`, enabling filtering like "only retrieve from Methods sections."

### Similarity Threshold Filtering

By default, Qdrant returns the top K results regardless of relevance score. For a research chatbot, low-scoring results can introduce noise and mislead the LLM. A similarity threshold filters these out.

Cosine similarity scores range from 0 (unrelated) to 1 (identical). For academic RAG, a threshold of **0.45–0.55** is a practical starting point.

```python
def retrieve_with_threshold(
    query: str,
    collection: str,
    top_k: int = 5,
    threshold: float = 0.50
) -> list[dict]:
    query_vector = embedder.encode(query).tolist()
    results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=threshold   # Qdrant native threshold support
    )
    return [
        {
            "text":   r.payload.get("text", ""),
            "pmcid":  r.payload.get("pmcid", r.payload.get("filename", "")),
            "title":  r.payload.get("title", ""),
            "score":  round(r.score, 3),
        }
        for r in results
    ]
```

If the query is too niche and nothing clears the threshold, the chatbot should respond with "no relevant literature found" rather than hallucinating from low-quality chunks.

**Threshold guidance:**

| Score Range | Interpretation |
|---|---|
| > 0.70 | Strong match — highly relevant |
| 0.50–0.70 | Good match — likely relevant |
| 0.35–0.50 | Weak match — use with caution |
| < 0.35 | Poor match — likely noise, discard |

---

## Configuration

All pipeline parameters are managed in `config.ini`:

```ini
[NCBI]
search_url = https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
fetch_url  = https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
sleep      = 0.4                 # seconds between NCBI requests (rate limiting)
search_limit_per_query = 200     # max papers per query before dedup

[S3]
base_url = https://pmc-oa-opendata.s3.amazonaws.com
sleep    = 0.3                   # seconds between S3 requests

[CHUNKING]
chunk_size    = 400              # words per chunk (~500 tokens)
chunk_overlap = 50               # word overlap between consecutive chunks

[EMBEDDING]
model      = BAAI/bge-base-en-v1.5
dim        = 768
batch_size = 32

[QDRANT]
host            = localhost
port            = 6333
collection_name = nih_stage_model
```

Search queries are defined directly in `nih_rag_pipeline.py` as Python strings (not in the `.ini`) due to `configparser` limitations with multi-line complex values.

---

## RAG Test

A lightweight test script (`rag_test.py`) validates the full pipeline with five queries that map directly to NIH Stage Model stages 0–IV:

```python
# Stage 0 — Basic science / mechanism exploration
rag_query("I am studying whether reading confidence-building books reduces anxiety during cancer treatment.")

# Stage I — Pilot and feasibility
rag_query("We designed a 6-week program and are running a small pilot study with 20 patients.")

# Stage II — Randomized controlled trial
rag_query("We conducted an RCT with 300 cancer patients comparing our intervention to standard care.")

# Stage III — Effectiveness in real-world settings
rag_query("Our intervention has shown positive results and we are testing it across multiple oncology clinics.")

# Stage IV — Dissemination and implementation
rag_query("Our program is now being implemented across hospital systems nationwide.")
```

Run:
```bash
# Start Ollama first
ollama serve

# In a separate terminal
python rag_test.py
```

Expected output per query:
```
Query: We designed a 6-week program...
--------------------------------------------------
Retrieved 5 chunks:
  [0.713] PMC8234521 — A pilot feasibility study of a Stage I behavioral...
  [0.681] PMC7891234 — Developing and piloting a reading-based intervention...
  ...

Generating response...

Response:
Based on the retrieved literature, your study design aligns with Stage I
of the NIH Stage Model, which focuses on...
```

---

## Project Structure

```
NIH_RAG_Project/
├── config.ini                  # All pipeline configuration
├── nih_rag_pipeline.py         # PMC eSearch → S3 → Qdrant pipeline
├── pdf_ingest_pipeline.py      # Local PDF → Qdrant pipeline
├── rag_test.py                 # RAG query test script
├── paper_files/                # Local PDF case studies
├── qdrant_storage/             # Qdrant persistent storage (Docker volume)
└── .env                        # NCBI_EMAIL environment variable
```

---

## Requirements

```bash
pip install requests python-dotenv sentence-transformers qdrant-client pdfplumber
```

Ollama: [ollama.com](https://ollama.com)

```bash
ollama pull qwen2.5:3b-instruct
```

---

## Environment Variables

Create a `.env` file in the project root:

```
NCBI_EMAIL=your_email@example.com
```

This is required by NCBI's API terms of use for rate limit identification. No API key needed — NCBI's eSearch and PMC's S3 bucket are both publicly accessible without authentication.

---

## Notes on PMC Data Access

PMC is transitioning its distribution infrastructure in 2026. The FTP service and OA Web Service API are being retired in August 2026. This pipeline uses the updated S3-based access pattern (`pmc-oa-opendata.s3.amazonaws.com`) which is the forward-compatible approach and will continue to work post-transition. No AWS account or credentials are required — the bucket is world-readable via `--no-sign-request`.
