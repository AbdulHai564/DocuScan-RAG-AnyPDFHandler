import sys
import types
fake = types.ModuleType('langchain_community.chat_models.vertexai')
fake.ChatVertexAI = None
sys.modules['langchain_community.chat_models.vertexai'] = fake

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retrieval import sementic_retrieval, bm25_retrieval, hybrid_retrieval, reranker, fetch_parents
from llm import call_llm
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from ragas.run_config import RunConfig


import time
time.sleep(2)


run_config = RunConfig(max_workers=1, timeout=180,max_retries=3)

ragas_llm = LangchainLLMWrapper(ChatGroq(model="openai/gpt-oss-20b"), run_config=RunConfig(max_retries=1))
ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
    metric.llm = ragas_llm
    metric.embeddings = ragas_embeddings

eval_data = [
    {"question": "What was NVIDIA's total revenue for fiscal year 2023?",
     "ground_truth": "NVIDIA's total revenue for fiscal year 2023 was $26.97 billion."},

    {"question": "What was NVIDIA's net income for fiscal year 2023?",
     "ground_truth": "NVIDIA's net income for fiscal year 2023 was $4,368 million."},

    {"question": "What was NVIDIA's Data Center revenue for fiscal year 2023?",
     "ground_truth": "NVIDIA's Data Center revenue was $15,010 million, up 41% from the prior year."},

    {"question": "What was NVIDIA's gross margin for fiscal year 2023?",
     "ground_truth": "NVIDIA's gross margin for fiscal year 2023 was 56.9%."},

    {"question": "Why was the Arm acquisition terminated?",
     "ground_truth": "The Arm acquisition was terminated due to significant regulatory challenges preventing the completion of the transaction."},

    {"question": "How much did NVIDIA return to shareholders in fiscal year 2023?",
     "ground_truth": "NVIDIA returned $10.44 billion to shareholders in fiscal year 2023 through share repurchases and cash dividends."},
]
questions, answers, contexts, ground_truths = [], [], [], []

semantic = sementic_retrieval()
bm25 = bm25_retrieval()

for item in eval_data:
    q = item["question"]
    hybrid_docs = hybrid_retrieval(semantic, bm25, q)
    reranked = reranker(q, hybrid_docs)
    parents = fetch_parents(reranked)
    context_texts = [doc.page_content for doc in parents]
    answer = call_llm(parents, q)
    questions.append(q)
    answers.append(answer)
    contexts.append(context_texts)
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=run_config,
 
)

print("\n========== RAGAS SCORES ==========")
print(results)
print("===================================")
print(results.to_pandas())