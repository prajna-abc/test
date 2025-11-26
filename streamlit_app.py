import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "papers"
EMBED_MODEL = "text-embedding-3-small"


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


def get_clients() -> Tuple[QdrantClient, OpenAI]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
        st.stop()
    qdrant_host = os.getenv("QDRANT_HOST", "")
    qdrant_port = os.getenv("QDRANT_PORT", "")
    qdrant_url = os.getenv("QDRANT_URL", "")
    if qdrant_url:
        qdrant = QdrantClient(url=qdrant_url)
    elif qdrant_host:
        qdrant = QdrantClient(host=qdrant_host, port=int(qdrant_port or 6333))
    else:
        qdrant = QdrantClient(path="qdrant_local")
    client = OpenAI(api_key=api_key)
    ensure_collection(qdrant, vector_size=1536)
    return qdrant, client


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
def chat_page(df: pd.DataFrame, qdrant: QdrantClient, oa_client: OpenAI, meta_map: dict):
    st.subheader("Chat")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    scope = st.radio("Scope", ["Full corpus", "Single paper"], horizontal=True)
    uuid_filter = ""
    if scope == "Single paper":
        options = [(row["paper_title"], row["uuid"]) for _, row in df.iterrows() if pd.notna(row.get("uuid"))]
        selection = st.selectbox("Select paper", options=options, format_func=lambda x: x[0] if isinstance(x, tuple) else x)
        if selection:
            _, uuid_filter = selection
    top_k = st.slider("Results to retrieve", 3, 10, 5)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about the papers...")
    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        hits = search_chunks(qdrant, oa_client, user_q, top_k=top_k, uuid_filter=uuid_filter)
        if not hits:
            answer = "No results retrieved from the vector store."
        else:
            meta_map_use = {uuid_filter: meta_map.get(uuid_filter, {})} if uuid_filter else meta_map
            answer = answer_query(oa_client, user_q, hits, meta_map_use)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        if hits:
            with st.expander("Context (top hits)"):
                for h in hits:
                    meta = meta_map.get(h.get("uuid", ""), {})
                    st.markdown(f"- UUID: {h.get('uuid','')} | Title: {meta.get('paper_title','')} | Year: {meta.get('year','')} | Score: {h.get('score',0):.3f}")
                    st.write(h.get("chunk", ""))
def browse_page(df: pd.DataFrame):
    st.subheader("Browse papers")
    col1, col2, col3 = st.columns(3)
    with col1:
        modality_options = sorted(set(val.strip() for val in ",".join(df["camera_modality"].fillna("")).split(",") if val.strip()))
        modality_filter = st.multiselect("Camera modality", modality_options, modality_options)
    with col2:
        impairment_options = sorted(set(val.strip() for val in ",".join(df["impairment_type"].fillna("")).split(",") if val.strip()))
        impairment_filter = st.multiselect("Impairment type", impairment_options, impairment_options)
    with col3:
        relevance_options = sorted(df["relevance_score"].dropna().unique().tolist())
        relevance_filter = st.multiselect("Relevance", relevance_options, relevance_options)

    def row_matches(row) -> bool:
        def match_list(field_val: str, selected: list[str]) -> bool:
            items = [p.strip() for p in str(field_val).split(",") if p.strip()]
            return any(i in selected for i in items) if selected else True
        return (
            match_list(row["camera_modality"], modality_filter)
            and match_list(row["impairment_type"], impairment_filter)
            and ((row["relevance_score"] in relevance_filter) if relevance_filter else True)
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
    df = load_analysis()
    qdrant, oa_client = get_clients()
    meta_map = metadata_by_uuid(df)

    st.set_page_config(page_title="Intoxication Vision Papers", layout="wide")
    st.title("Vision-Based Intoxication Detection Papers")
    st.caption("Browse + chat on one page; CSV view on another.")

    page = st.sidebar.radio("Navigate", ["Main", "CSV"])

    if page == "Main":
        left, right = st.columns(2)
        with left:
            browse_page(df)
        with right:
            chat_page(df, qdrant, oa_client, meta_map)
    elif page == "CSV":
        st.subheader("CSV View (analysis.csv)")
        cols = ["url", "paper_title", "summary", "year"]
        available_cols = [c for c in cols if c in df.columns]
        df_view = df[available_cols].copy()
        if "url" in df_view.columns:
            df_view["url"] = df_view["url"].apply(lambda x: f"[link]({x})" if pd.notna(x) and str(x).strip() else "")
        st.markdown(df_view.to_markdown(index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
