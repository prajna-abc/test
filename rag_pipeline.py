import csv
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DEFAULT_COLLECTION = "papers"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200
PDF_DIR = "vision-based pdfs"
CSV_PATH = "pdfs.csv"  # expects url, uuid headers (case-insensitive)


@dataclass
class PaperEntry:
    url: str
    uuid: str


def load_entries(csv_path: str = CSV_PATH) -> List[PaperEntry]:
    entries: List[PaperEntry] = []
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return entries
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lowered = {k.lower(): v for k, v in row.items()} if row else {}
            url = (row.get("url") or lowered.get("url") or "").strip()
            uuid = (row.get("uuid") or lowered.get("uuid") or "").strip()
            if not uuid:
                continue
            entries.append(PaperEntry(url=url, uuid=uuid))
    return entries


def read_pdf(entry: PaperEntry) -> str:
    filename = entry.uuid if entry.uuid.lower().endswith(".pdf") else f"{entry.uuid}.pdf"
    path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(path):
        print(f"PDF missing: {path}")
        return ""
    try:
        reader = PdfReader(path)
        texts = []
        for idx, page in enumerate(reader.pages, start=1):
            try:
                texts.append(page.extract_text() or "")
            except Exception as exc:
                print(f"Warn: failed page {idx} for {path}: {exc}")
        return "\n".join(texts)
    except Exception as exc:
        print(f"Error reading {path}: {exc}")
        return ""


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    """Yield chunks lazily to reduce memory usage."""
    if not text:
        return
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        yield text[start:end]
        if end == length:
            break
        start = end - overlap
        if start < 0:
            start = 0


def get_client_and_openai() -> tuple[QdrantClient, OpenAI, str]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    client = OpenAI(api_key=api_key)
    if qdrant_url:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
    else:
        qdrant = QdrantClient(path="qdrant_local")
    return qdrant, client, api_key


def ensure_collection(qdrant: QdrantClient, collection_name: str = DEFAULT_COLLECTION, vector_size: int = 1536):
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def embed_texts(
    client: OpenAI, texts: List[str], model: str = DEFAULT_EMBED_MODEL, batch_size: int = 32
) -> List[List[float]]:
    """Embed texts in small batches to avoid OOM."""
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings


def upsert_entries(
    qdrant: QdrantClient,
    client: OpenAI,
    entries: Iterable[PaperEntry],
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 8,
    max_chunks_per_doc: int = 2000,
):
    ensure_collection(qdrant, collection_name)
    for entry in entries:
        text = read_pdf(entry)
        if not text:
            continue
        total = 0
        batch_chunks: List[str] = []
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            if max_chunks_per_doc and idx >= max_chunks_per_doc:
                print(f"Reached max_chunks_per_doc ({max_chunks_per_doc}) for {entry.uuid}")
                break
            batch_chunks.append(chunk)
            if len(batch_chunks) >= batch_size:
                embeddings = embed_texts(client, batch_chunks, model=embed_model, batch_size=batch_size)
                points = []
                base = idx - len(batch_chunks) + 1
                for j, (chunk_text_val, vector) in enumerate(zip(batch_chunks, embeddings)):
                    global_idx = base + j
                    points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "uuid": entry.uuid,
                            "url": entry.url,
                            "chunk": chunk_text_val,
                            },
                        )
                    )
                qdrant.upsert(collection_name=collection_name, points=points)
                total += len(points)
                batch_chunks = []
        # handle remainder
        if batch_chunks:
            embeddings = embed_texts(client, batch_chunks, model=embed_model, batch_size=batch_size)
            points = []
            base = total
            for j, (chunk_text_val, vector) in enumerate(zip(batch_chunks, embeddings)):
                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "uuid": entry.uuid,
                            "url": entry.url,
                            "chunk": chunk_text_val,
                    },
                )
            )
            qdrant.upsert(collection_name=collection_name, points=points)
            total += len(points)
        print(f"Indexed {total} chunks for {entry.uuid}")


def query(
    qdrant: QdrantClient,
    client: OpenAI,
    query_text: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[dict]:
    vector = embed_texts(client, [query_text], model=embed_model)[0]
    res = qdrant.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    hits = res.points if hasattr(res, "points") else res
    results = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        score = getattr(h, "score", 0)
        results.append(
            {
                "uuid": payload.get("uuid", ""),
                "url": payload.get("url", ""),
                "chunk": payload.get("chunk", ""),
                "score": score,
            }
        )
    return results


def main():
    """CLI entry: index all PDFs from CSV into Qdrant."""
    load_dotenv()
    qdrant, client, _ = get_client_and_openai()
    entries = load_entries(CSV_PATH)
  
    if not entries:
        print("No entries found in CSV.")
        return
    upsert_entries(qdrant, client, entries)


if __name__ == "__main__":
    main()
