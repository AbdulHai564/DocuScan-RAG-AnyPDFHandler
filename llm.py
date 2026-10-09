from config import *
from langchain_groq import ChatGroq
from retrieval import fetch_parents



GROQ_CLIENT=ChatGroq(model="openai/gpt-oss-120b")


def call_llm(results,query):

  context = "\n\n".join([doc.page_content for doc in results])

  prompt=f"""Answer the following question on the context below.I need you to read the context properly before answering

Context:{context}
question:{query}
"""
  answer=GROQ_CLIENT.invoke(prompt)

  return answer.content


