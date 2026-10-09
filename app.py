import streamlit as st
import uuid
from ingest import load_chunk, store
from retrieval import sementic_retrieval, bm25_retrieval, hybrid_retrieval, reranker, fetch_parents
from llm import call_llm

st.set_page_config(page_title="DocuScan",layout="centered")

st.title("🔍 DocuScan")
st.caption("Upload any PDF — scanned, messy, or clean — and ask questions about it.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

with st.sidebar:
    st.header(" Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and (not st.session_state.pdf_ready or uploaded_file.name != st.session_state.pdf_name):
        with st.spinner("Processing PDF..."):
            path = f"/tmp/{st.session_state.session_id}_{uploaded_file.name}"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            parent_chunks, child_chunks = load_chunk(path)
            store(parent_chunks, child_chunks)
            st.session_state.pdf_ready = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
        st.success("PDF ready! Ask away.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.pdf_ready:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                semantic = sementic_retrieval()
                bm25 = bm25_retrieval()
                hybrid = hybrid_retrieval(semantic, bm25, prompt)
                reranked = reranker(prompt, hybrid)
                parents = fetch_parents(reranked)
                answer = call_llm(parents, prompt)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})