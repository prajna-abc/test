import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

COLLECTION_NAME = "papers"
EMBED_MODEL = "text-embedding-3-small"
AUTH_COOKIE_NAME = "abclabs_auth"
AUTH_COOKIE_KEY = "abclabs_auth_key"
AUTH_COOKIE_DURATION = 1  # days


# ------------ Data loading ------------
def load_analysis(path: str = "analysis.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"analysis.csv not found at {os.path.abspath(path)}")
        st.stop()
    df = pd.read_csv(path)
    def parse_details(val: str):
        try:
            return json.loads(val) if isinstance(val, str) else {}
        except json.JSONDecodeError:
            return {"raw": val}
    df["details_dict"] = df["details_json"].apply(parse_details)
    return df


def metadata_by_uuid(df: pd.DataFrame) -> Dict[str, dict]:
    return {str(row["uuid"]): row for _, row in df.iterrows() if pd.notna(row.get("uuid"))}


# ------------ Clients ------------
def ensure_collection(qdrant: QdrantClient, vector_size: int = 1536):
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="uuid",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        # Index likely already exists; ignore errors to keep startup fast.
        pass


def get_clients() -> Tuple[QdrantClient, OpenAI]:
    """Prefer .env/environment; only touch st.secrets if env is missing."""
    from pathlib import Path

    def load_env_and_get():
        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(env_path)
        return (
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("QDRANT_URL", ""),
            os.getenv("QDRANT_API_KEY", ""),
        )

    # First try env/.env so we don't touch st.secrets when it's absent
    api_key, qdrant_url, qdrant_api_key = load_env_and_get()

    # If still missing, try st.secrets but guard against missing secrets.toml
    if not api_key or not qdrant_url:
        try:
            secrets = st.secrets  # may raise StreamlitSecretNotFoundError if file absent
        except Exception:
            secrets = {}
        if secrets:
            api_key = secrets.get("OPENAI_API_KEY", api_key)
            qdrant_url = secrets.get("QDRANT_URL", qdrant_url)
            qdrant_api_key = secrets.get("QDRANT_API_KEY", qdrant_api_key)

    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add it to .streamlit/secrets.toml or your .env/environment.")
        st.stop()
    if not qdrant_url:
        st.error("QDRANT_URL is missing. Add it to .streamlit/secrets.toml or your .env/environment.")
        st.stop()

    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
    client = OpenAI(api_key=api_key)

    ensure_collection(qdrant, vector_size=1536)
    return qdrant, client


# ------------ Auth ------------
def load_authenticator():
    # Use secrets if present; fall back to env vars or disable auth locally when absent.
    try:
        auth_cfg = st.secrets.get("auth", {}) if hasattr(st, "secrets") else {}
    except Exception:
        auth_cfg = {}
    if not auth_cfg:
        # Local dev fallback: allow access when no secrets.toml is present.
        st.info("Auth is disabled because no .streamlit/secrets.toml was found. Set [auth] in secrets to enable login.")
        return None, None, None

    credentials = auth_cfg.get("credentials", {})
    allowed_domain = auth_cfg.get("allowed_domain", "")
    if not credentials or not allowed_domain:
        st.error(
            "Authentication not configured. Set [auth.credentials] and [auth.allowed_domain] in .streamlit/secrets.toml."
        )
        st.stop()
    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name=AUTH_COOKIE_NAME,
        key=AUTH_COOKIE_KEY,
        cookie_expiry_days=AUTH_COOKIE_DURATION,
    )
    name, auth_status, username = authenticator.login("Login", "main")
    if auth_status:
        if not username or not username.endswith(f"@{allowed_domain}"):
            st.error("Access restricted to authorized ABC Labs accounts.")
            st.stop()
    elif auth_status is False:
        st.error("Invalid credentials.")
        st.stop()
    else:
        st.stop()
    return authenticator, name, username


# ------------ Retrieval + QA ------------
def search_chunks(qdrant: QdrantClient, client: OpenAI, query: str, top_k: int = 5, uuid_filter: str = "") -> List[dict]:
    emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    kwargs = {"collection_name": COLLECTION_NAME, "query": emb, "limit": top_k, "with_payload": True}
    if uuid_filter:
        kwargs["query_filter"] = {"must": [{"key": "uuid", "match": {"value": uuid_filter}}]}
    try:
        res = qdrant.query_points(**kwargs)
    except Exception as exc:
        st.error(f"Qdrant query failed: {exc}")
        return []
    hits = res.points if hasattr(res, "points") else res
    results = []
    for h in hits or []:
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


def answer_query(client: OpenAI, question: str, contexts: List[dict], meta_map: dict) -> str:
    context_blocks = []
    for ctx in contexts:
        meta = meta_map.get(ctx.get("uuid", ""), {})
        title = meta.get("paper_title", "")
        year = meta.get("year", "")
        cite = f"[Paper: {title}, Year: {year}]" if title or year else ""
        block = f"{cite}\n{ctx.get('chunk','')}"
        context_blocks.append(block)
    context_text = "\n\n".join(context_blocks)
    system_prompt = (
        "You are an expert AI literature assistant specialized in computer vision, biometrics, intoxication detection, "
        "and human impairment analysis. Answer strictly from the provided context. "
        'If the answer is missing, reply: "The retrieved documents do not contain this information." '
        "Cite facts as [Paper: <paper_title>, Year: <year>]."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]
    resp = client.chat.completions.create(model="gpt-4.1", messages=messages, temperature=0)
    return resp.choices[0].message.content.strip()


# ------------ UI ------------
def browse_page(df: pd.DataFrame):
    st.subheader("Browse papers")
    modality_options = sorted(
        set(val.strip() for val in ",".join(df["camera_modality"].fillna("")).split(",") if val.strip())
    )
    select_all = st.checkbox("Select all camera modalities", value=True)
    modality_filter = st.multiselect(
        "Camera modality",
        modality_options,
        modality_options if select_all else [],
        placeholder="Choose one or more modalities",
    )
    feature_options = sorted(
        set(val.strip() for val in ",".join(df["visual_feature_types"].fillna("")).split(",") if val.strip())
    )
    select_all_feat = st.checkbox("Select all visual feature types", value=True)
    feature_filter = st.multiselect(
        "Visual feature types",
        feature_options,
        feature_options if select_all_feat else [],
        placeholder="Choose one or more feature types",
    )

    def row_matches(row) -> bool:
        def match_list(field_val: str, selected: list[str]) -> bool:
            items = [p.strip() for p in str(field_val).split(",") if p.strip()]
            return any(i in selected for i in items) if selected else True
        return match_list(row["camera_modality"], modality_filter) and match_list(
            row["visual_feature_types"], feature_filter
        )

    filtered = df[df.apply(row_matches, axis=1)].copy()
    st.write(f"Showing {len(filtered)} of {len(df)} papers.")
    for _, row in filtered.iterrows():
        with st.expander(f"{row.get('paper_title','Untitled')} | {row.get('year','')} | {row.get('relevance_score','')}"):
            st.write(f"**Summary:** {row.get('summary','')}")
            core_cols = ["camera_modality", "visual_feature_types", "impairment_type", "real_time_capability", "sample_size", "environment", "primary_accuracy_metric", "relevance_reason"]
            core_display = {k: row.get(k, "") for k in core_cols}
            st.write("**Core fields**")
            st.json(core_display, expanded=False)
            st.write("**Details**")
            st.json(row.get("details_dict", {}), expanded=False)
            pdf_link = row.get("url", "")
            st.markdown(f"[Open PDF]({pdf_link})" if pdf_link else "No PDF link available.")


def chat_page(df: pd.DataFrame, qdrant: QdrantClient, oa_client: OpenAI, meta_map: dict):
    st.subheader("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def maybe_answer_simple_question(question: str, uuid_filter: str) -> str | None:
        """Handle simple corpus-level facts without hitting the vector store."""
        q = question.lower()
        count_phrases = ["how many papers", "how many documents", "number of papers", "count of papers"]
        if any(p in q for p in count_phrases):
            if uuid_filter:
                return "Only the selected paper is in scope, so the count is 1."
            return f"There are {len(df)} papers in the corpus."
        return None

    scope = st.radio("Chat scope", ["Full corpus", "Single paper"], horizontal=True)
    uuid_filter = ""
    if scope == "Single paper":
        paper_options = [(row["paper_title"], row["uuid"]) for _, row in df.iterrows() if pd.notna(row.get("uuid"))]
        selection = st.selectbox("Select paper", options=paper_options, format_func=lambda x: x[0] if isinstance(x, tuple) else x)
        if selection:
            _, uuid_filter = selection

    top_k = st.slider("Results to retrieve", 3, 10, 5)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about the papers...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        simple_answer = maybe_answer_simple_question(user_q, uuid_filter)
        if simple_answer:
            st.session_state.messages.append({"role": "assistant", "content": simple_answer})
            with st.chat_message("assistant"):
                st.markdown(simple_answer)
            return

        hits = search_chunks(qdrant, oa_client, user_q, top_k=top_k, uuid_filter=uuid_filter)
        if not hits:
            answer = "No results retrieved from the vector store."
        else:
            if uuid_filter:
                meta_map_use = {uuid_filter: meta_map.get(uuid_filter, {})}
            else:
                meta_map_use = meta_map
            answer = answer_query(oa_client, user_q, hits, meta_map_use)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        if hits:
            with st.expander("Context (top hits)"):
                for h in hits:
                    meta = meta_map.get(h.get("uuid", ""), {})
                    st.markdown(f"- UUID: {h.get('uuid','')} | Title: {meta.get('paper_title','')} | Year: {meta.get('year','')} | Score: {h.get('score',0):.3f}")
                    st.write(h.get("chunk", ""))


def main():
    st.set_page_config(page_title="Intoxication Vision Papers", layout="wide")

    df = load_analysis()
    qdrant, oa_client = get_clients()
    meta_map = metadata_by_uuid(df)

    authenticator, name, username = load_authenticator()

    st.title("Vision-Based Intoxication Detection Papers")
    st.caption("Browse and chat in one place.")

    if authenticator and name:
        with st.sidebar:
            st.write(f"Signed in as **{name}**")
            authenticator.logout("Logout", "sidebar")

    left, right = st.columns(2)
    with left:
        browse_page(df)
    with right:
        chat_page(df, qdrant, oa_client, meta_map)


if __name__ == "__main__":
    main()
