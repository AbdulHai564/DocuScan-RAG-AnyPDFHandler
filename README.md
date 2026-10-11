# DocuScan — OCR-Powered RAG for Any PDF

> Ask questions against any PDF — scanned or digital — with hybrid retrieval, reranking, and a fast LLM response.

**Live Demo:** [docuscan-rag-anypdfhandler-ctmsbsg8qwyp3nudfapzwx.streamlit.app](https://docuscan-rag-anypdfhandler-ctmsbsg8qwyp3nudfapzwx.streamlit.app)  
**Repo:** [github.com/abdulhai564/docuscan-rag-anypdfhandler](https://github.com/abdulhai564/docuscan-rag-anypdfhandler)

---

## What It Does

Most RAG pipelines break on scanned PDFs — they extract blank pages and silently return garbage. DocuScan detects scanned pages automatically and runs Tesseract OCR as a fallback, so the pipeline works on any PDF regardless of how it was created.

---

## Pipeline

```
PDF Upload
    │
    ▼
PyMuPDF text extraction
    │
    ├── Digital page? → extract text directly
    └── Scanned page? → Tesseract OCR fallback
    │
    ▼
Parent / Child Chunking
    │
    ▼
Embeddings (all-MiniLM-L6-v2) → Qdrant Cloud
    │
    ▼
Hybrid Retrieval
  ├── BM25 keyword search    (30%)
  └── Semantic vector search (70%)
    │
    ▼
Cohere Rerank (rerank-v3.5)
    │
    ▼
Fetch Parent Chunks
    │
    ▼
Groq LLM (openai/gpt-oss-120b)
    │
    ▼
Answer
```

---

## Stack

| Component | Library / Service |
|---|---|
| PDF extraction | PyMuPDF |
| OCR | Tesseract + pytesseract + Pillow |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector store | Qdrant Cloud |
| Reranking | Cohere `rerank-v3.5` |
| LLM | Groq `openai/gpt-oss-120b` |
| Frontend | Streamlit Cloud |
| Orchestration | LangChain |

---

## Project Structure

```
docuscan-rag-anypdfhandler/
├── app.py          # Streamlit UI and main entry point
├── ingest.py       # PDF loading, OCR detection, chunking, Qdrant upload
├── retrieval.py    # Hybrid BM25 + semantic retrieval + Cohere rerank
├── llm.py          # Groq LLM call and prompt template
├── config.py       # API keys, model names, Qdrant config
├── packages.txt    # System deps for Streamlit Cloud (tesseract-ocr)
└── requirements.txt
```

---

## RAGAS Evaluation

Evaluated on 6 questions from the NVIDIA FY2023 Annual Report (10-K).  
Judge LLM: `Qwen2.5-Coder` via LM Studio (local).  
Framework: `ragas==0.3.9`

| Metric | Score | What It Measures |
|---|---|---|
| Faithfulness | **0.56** | Does the answer stick to what the retrieved context actually says? |
| Answer Relevancy | **0.57** | Is the answer on-topic for the question asked? |
| Context Precision | **0.30** | Are the top-ranked chunks actually the useful ones? |
| Context Recall | **0.70** | Does the retrieved context cover what's needed to answer? |

> **Note:** Scores were produced with a code-specialized judge model. A general-purpose instruct model would yield more reliable faithfulness and relevancy scores. Evaluation is planned with a larger question set and a better judge model in the next iteration.

---

## What the Scores Mean

**Context Recall at 0.70** is the pipeline's strongest number — it retrieves the right information most of the time. The weakest metric is **Context Precision at 0.30**, meaning irrelevant chunks are making it into the top results alongside the useful ones. That is the primary area for improvement.

---

## Roadmap for Better Scores

**1. Fix context precision (biggest lever)**  
The retriever is fetching the right answer but ranking irrelevant chunks above it. Tuning the BM25/semantic weight ratio (currently 30/70) and experimenting with chunk size could push precision significantly.

**2. Clean table text at ingest time**  
Financial tables extracted from PDFs contain noisy whitespace and broken numbers (e.g. `$ \n26,974 \n`). A post-extraction cleaning step before chunking would improve both embedding quality and BM25 keyword matching.

**3. Better judge model**  
Rerunning eval with a general-purpose instruct model (e.g. `llama-3.3-70b` on Groq) would give more reliable faithfulness and relevancy scores, especially on financial question-answering tasks.

**4. Larger eval set**  
6 questions is too small — one bad answer skews all averages. A 15–20 question set with a mix of factual lookups, table-heavy questions, and multi-hop questions would give stable metrics.

**5. Metadata filtering**  
Tag chunks with page number and section at ingest time, and filter by section at query time (e.g. skip audit committee boilerplate when answering financial questions).

---

## Key Engineering Decisions

**EasyOCR → Tesseract:** EasyOCR crashed Streamlit Cloud on cold start due to memory limits. Tesseract runs headlessly with a single `packages.txt` entry and is stable on the free tier.

**Batched Qdrant uploads:** Uploading all chunks in one request hit Qdrant's write timeout on large PDFs. Switched to batches of 50 chunks with a retry loop.

**Parent/child chunking:** Small child chunks (256 tokens) are used for retrieval precision. Their parent chunks (1024 tokens) are fetched and sent to the LLM for fuller context.

---

## Running Locally

```bash
git clone https://github.com/abdulhai564/docuscan-rag-anypdfhandler
cd docuscan-rag-anypdfhandler
pip install -r requirements.txt

# Add keys to .env
GROQ_API_KEY=...
COHERE_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...

streamlit run app.py
```

**Tesseract must be installed separately:**
- Ubuntu/WSL: `sudo apt install tesseract-ocr`
- Windows: [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Author

**Abdulhai Adnan** — BS Data Science, Air University Karachi  
Building RAG pipelines and AI agents. Open to remote freelance projects.  
[GitHub](https://github.com/AbdulHai564)
