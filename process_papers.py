import csv
import io
import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# API key is loaded from environment (e.g., .env or Streamlit secrets).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# List of PDF URLs to process.
PDF_URLS = [
    "",
]

PDF_CSV_PATH = "pdfs.csv"  # expects columns: url, uuid (uuid matches PDF filename on Drive)
PDF_DIR = "vision-based pdfs"  # local folder containing PDFs named by uuid
PROCESS_LIMIT = None # limit number of PDFs to process (set None to process all)

MODEL = "gpt-4.1"
SYSTEM_PROMPT = """You are a research assistant specializing in computer vision for real-time intoxication detection.

Your task is to analyze scientific papers and return a structured JSON output using a hybrid schema:

1. “core” section → must always follow the fixed standardized fields.
2. “details” section → flexible; include any rich paper-specific information extracted from the PDF.

RULES:
- Only include information relevant to camera-based intoxication/impairment detection (RGB, IR, NIR, Thermal, Hyperspectral, Depth).
- If the paper includes non-visual methods, ignore them or mention them only in limitations.
- Output ONLY valid JSON that respects the schema below.
- Do NOT invent fields outside of the "core" schema. Extra fields must go inside "details".

THE HYBRID SCHEMA:

{
  "core": {
    "paper_title": "",
    "year": "",
    "source": "",
    "camera_modality": [],
    "visual_feature_types": [],
    "impairment_type": [],
    "real_time_capability": "",
    "sample_size": "",
    "environment": "",
    "primary_accuracy_metric": "",
    "relevance_score": "",
    "relevance_reason": ""
  },
  "details": {}
}

Definitions:
- "camera_modality": RGB, IR/NIR, Thermal, Depth, Hyperspectral
- "visual_feature_types": e.g., eye-movement, pupil dynamics, facial micro-expressions, head pose, body sway, tremor, thermal-face-patterns, cognitive-visual behaviours
- "impairment_type": alcohol, cannabis, opioids, general impairment, fatigue, mixed
- "relevance_score": ⭐, ⭐⭐, ⭐⭐⭐ where ⭐⭐⭐ = highly relevant real-time visual intoxication detection

Always follow this schema strictly.
Provide a separate short summary field (not inside details)."""

JSON_SCHEMA = {
    "summary": "",
    "core": {
        "paper_title": "",
        "year": "",
        "source": "",
        "camera_modality": [],
        "visual_feature_types": [],
        "impairment_type": [],
        "real_time_capability": "",
        "sample_size": "",
        "environment": "",
        "primary_accuracy_metric": "",
        "relevance_score": "",
        "relevance_reason": "",
    },
    "details": {},
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_schema",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "core": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "paper_title": {"type": "string"},
                        "year": {"type": "string"},
                        "source": {"type": "string"},
                        "camera_modality": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "visual_feature_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "impairment_type": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "real_time_capability": {"type": "string"},
                        "sample_size": {"type": "string"},
                        "environment": {"type": "string"},
                        "primary_accuracy_metric": {"type": "string"},
                        "relevance_score": {"type": "string"},
                        "relevance_reason": {"type": "string"},
                    },
                    "required": [
                        "paper_title",
                        "year",
                        "source",
                        "camera_modality",
                        "visual_feature_types",
                        "impairment_type",
                        "real_time_capability",
                        "sample_size",
                        "environment",
                        "primary_accuracy_metric",
                        "relevance_score",
                        "relevance_reason",
                    ],
                },
                "details": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "required": ["summary", "core", "details"],
        },
        "strict": False,
    },
}


def load_pdf_entries(csv_path: str) -> list[dict]:
    """Load entries from local CSV; expects columns: url, uuid (case-insensitive)."""
    entries: list[dict] = []
    try:
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return entries
        csvfile = open(csv_path, newline="", encoding="utf-8")
        with csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # case-insensitive access
                lowered = {k.lower(): v for k, v in row.items()} if row else {}
                url = (row.get("url") or lowered.get("url") or "").strip()
                uuid = (row.get("uuid") or lowered.get("uuid") or "").strip()
                if not url and not uuid:
                    continue
                entries.append({"url": url, "uuid": uuid})
    except Exception as exc:
        print(f"Error reading CSV {csv_path}: {exc}")
    return entries


def read_pdf_from_disk(uuid: str) -> Optional[bytes]:
    """Load a PDF from the local folder using uuid as the filename (with .pdf)."""
    if not uuid:
        print("Missing uuid for PDF.")
        return None
    filename = f"{uuid}.pdf" if not uuid.lower().endswith(".pdf") else uuid
    path = os.path.join(PDF_DIR, filename)
    print(f"Reading PDF from: {path}")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as exc:
        print(f"Error reading {path}: {exc}")
        return None


def extract_text(pdf_bytes: bytes) -> str:
    """Extract full text from PDF bytes (no truncation)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:
            print(f"Warning: failed to extract page {index}: {exc}")
    return "\n".join(pages_text)


def call_openai(client: OpenAI, text: str) -> Optional[dict]:
    """Send text to OpenAI ChatCompletion and parse JSON response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            response_format=RESPONSE_FORMAT,
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            print(f"JSON parse error: {exc}")
            print("Raw response:")
            print(content)
            return None
    except Exception as exc:
        print(f"OpenAI API error: {exc}")
        return None


def url_to_filename(url: str, uuid: str = "") -> str:
    """Derive a JSON filename from uuid (preferred) or PDF URL."""
    if uuid:
        base = uuid[:-4] if uuid.lower().endswith(".pdf") else uuid
        return f"{base}.json"
    filename = url.rstrip("/").split("/")[-1]
    if filename.lower().endswith(".pdf"):
        filename = filename[:-4]
    if not filename:
        filename = "paper"
    return f"{filename}.json"


def save_json(data: dict, filename: str) -> None:
    """Save JSON data to a file in the current directory."""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    print(f"Saved: {filename}")


def write_analysis_csv(records: list[dict], filename: str = "analysis.csv") -> None:
    """Write a flat CSV summary of all processed papers."""
    fieldnames = [
        "url",
        "uuid",
        "file",
        "summary",
        "paper_title",
        "year",
        "source",
        "camera_modality",
        "visual_feature_types",
        "impairment_type",
        "real_time_capability",
        "sample_size",
        "environment",
        "primary_accuracy_metric",
        "relevance_score",
        "relevance_reason",
        "details_json",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            core = record.get("core", {})
            details = record.get("details", {})
            writer.writerow(
                {
                    "url": record.get("url", ""),
                    "uuid": record.get("uuid", ""),
                    "file": record.get("file", ""),
                    "summary": record.get("summary", ""),
                    "paper_title": core.get("paper_title", ""),
                    "year": core.get("year", ""),
                    "source": core.get("source", ""),
                    "camera_modality": ", ".join(core.get("camera_modality", []) or []),
                    "visual_feature_types": ", ".join(
                        core.get("visual_feature_types", []) or []
                    ),
                    "impairment_type": ", ".join(core.get("impairment_type", []) or []),
                    "real_time_capability": core.get("real_time_capability", ""),
                    "sample_size": core.get("sample_size", ""),
                    "environment": core.get("environment", ""),
                    "primary_accuracy_metric": core.get("primary_accuracy_metric", ""),
                    "relevance_score": core.get("relevance_score", ""),
                    "relevance_reason": core.get("relevance_reason", ""),
                    "details_json": json.dumps(details, ensure_ascii=False),
                }
            )
    print(f"Saved summary CSV: {filename}")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key:
        print("Please set OPENAI_API_KEY (environment or .env).")
        return

    entries = load_pdf_entries(PDF_CSV_PATH)
    if entries:
        print(f"Loaded {len(entries)} entries from {PDF_CSV_PATH}")
    else:
        entries = [{"url": u, "uuid": ""} for u in PDF_URLS if u]
        print(f"No entries loaded from CSV. Using default list of {len(entries)} URL(s).")
    if not entries:
        print("No entries to process.")
        return

    if PROCESS_LIMIT:
        entries = entries[:PROCESS_LIMIT]
        print(f"Processing first {len(entries)} entries (limit={PROCESS_LIMIT}).")

    client = OpenAI(api_key=api_key)

    processed_records: list[dict] = []
    for entry in entries:
        url = entry.get("url", "")
        uuid = entry.get("uuid", "")
        print(f"Processing URL: {url} (uuid: {uuid})")
        pdf_bytes = read_pdf_from_disk(uuid)
        if not pdf_bytes:
            print(f"Skipping {url} because PDF file was not found or unreadable.")
            continue

        text = extract_text(pdf_bytes)
        if not text.strip():
            print(f"No text extracted from {url}. Skipping.")
            continue

        result = call_openai(client, text)
        if result is None:
            print(f"Skipping save for {url} due to JSON parsing issue.")
            continue

        filename = url_to_filename(url, uuid)
        save_json(result, os.path.join(".", filename))
        processed_records.append(
            {
                "url": url,
                "uuid": uuid,
                "file": filename,
                **result,
            }
        )

    if processed_records:
        write_analysis_csv(processed_records)


if __name__ == "__main__":
    main()
