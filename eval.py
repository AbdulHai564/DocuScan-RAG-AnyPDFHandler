
import sys
import types

# Compatibility workaround for the installed RAGAS/LangChain versions
fake = types.ModuleType("langchain_community.chat_models.vertexai")
fake.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = fake

import ragas.llms.base

ragas.llms.base.is_multiple_completion_supported = lambda llm: False

from dotenv import load_dotenv
load_dotenv()

from config import *

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from datasets import Dataset

from retrieval import (
    sementic_retrieval,
    bm25_retrieval,
    hybrid_retrieval,
    reranker,
    fetch_parents,
)

from llm import call_llm


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

run_config = RunConfig(
    max_workers=1,
    timeout=300,
    max_retries=2,
)


# --------------------------------------------------
# 2. LM Studio: Local evaluation LLM
# --------------------------------------------------

local_model = ChatOpenAI(
    model="qwen2.5-coder-3b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    temperature=0.1,
    n=1,
    max_tokens=1024,
    timeout=300,
    max_retries=2,
)

ragas_llm = LangchainLLMWrapper(
    langchain_llm=local_model,
    bypass_n=True,
)


# --------------------------------------------------
# 3. Hugging Face embeddings
# --------------------------------------------------

hf_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    hf_embeddings
)


# --------------------------------------------------
# 4. Configure RAGAS metrics
# --------------------------------------------------

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

for metric in metrics:
    metric.llm = ragas_llm
    metric.embeddings = ragas_embeddings


# --------------------------------------------------
# 5. Evaluation questions and ground truths
# --------------------------------------------------

eval_data = [
    {
        "question": "What was NVIDIA's total revenue for fiscal year 2023?",
        "ground_truth": (
            "NVIDIA's total revenue for fiscal year 2023 "
            "was $26.97 billion."
        ),
    },
    {
        "question": "What was NVIDIA's net income for fiscal year 2023?",
        "ground_truth": (
            "NVIDIA's net income for fiscal year 2023 "
            "was $4,368 million."
        ),
    },
    {
        "question": "What was NVIDIA's Data Center revenue for fiscal year 2023?",
        "ground_truth": (
            "NVIDIA's Data Center revenue was $15,010 million, "
            "up 41% from the prior year."
        ),
    },
    {
        "question": "What was NVIDIA's gross margin for fiscal year 2023?",
        "ground_truth": (
            "NVIDIA's gross margin for fiscal year 2023 was 56.9%."
        ),
    },
    {
        "question": "Why was the Arm acquisition terminated?",
        "ground_truth": (
            "The Arm acquisition was terminated due to significant "
            "regulatory challenges preventing the completion "
            "of the transaction."
        ),
    },
    {
        "question": "How much did NVIDIA return to shareholders in fiscal year 2023?",
        "ground_truth": (
            "NVIDIA returned $10.44 billion to shareholders in "
            "fiscal year 2023 through share repurchases "
            "and cash dividends."
        ),
    },
]


# --------------------------------------------------
# 6. Run the DocuScan retrieval pipeline
# --------------------------------------------------

questions = []
answers = []
contexts = []
ground_truths = []

print("\nInitializing retrieval systems...")

semantic = sementic_retrieval()
bm25 = bm25_retrieval()

print("Retrieval systems initialized.")
print("\nGenerating answers for evaluation...\n")

for index, item in enumerate(eval_data, start=1):
    question = item["question"]

    print(f"[{index}/{len(eval_data)}] {question}")

    # Semantic + BM25 hybrid retrieval
    hybrid_docs = hybrid_retrieval(
        semantic,
        bm25,
        question,
    )

    # Rerank retrieved documents
    reranked_docs = reranker(
        question,
        hybrid_docs,
    )

    # Retrieve parent documents
    parent_docs = fetch_parents(reranked_docs)

    # Extract context text
    context_texts = [
        doc.page_content
        for doc in parent_docs
    ]

    # Generate the answer using your existing LLM function
    answer = call_llm(
        parent_docs,
        question,
    )

    # Store evaluation data
    questions.append(question)
    answers.append(str(answer))
    contexts.append(context_texts)
    ground_truths.append(item["ground_truth"])

    print(f"Answer: {answer}\n")


# --------------------------------------------------
# 7. Build the RAGAS evaluation dataset
# --------------------------------------------------

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)


# --------------------------------------------------
# 8. Evaluate the RAG pipeline
# --------------------------------------------------

print("\nStarting RAGAS evaluation with local Qwen model...")
print("Keep LM Studio running while evaluation is in progress.\n")

results = evaluate(
    dataset,
    metrics=metrics,
    run_config=run_config,
)


# --------------------------------------------------
# 9. Display evaluation results
# --------------------------------------------------

print("\n========== RAGAS SCORES ==========")
print(results)
print("===================================\n")

results_df = results.to_pandas()

print(results_df.to_string(index=False))

# Save results for later analysis
results_df.to_csv(
    "ragas_evaluation_results.csv",
    index=False,
)

print("\nResults saved to ragas_evaluation_results.csv")
