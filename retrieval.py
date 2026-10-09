from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.retrievers import BM25Retriever,EnsembleRetriever
from langchain_cohere import CohereRerank
from config import *
from qdrant_client import QdrantClient,models
from langchain_core.documents import Document


embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client=QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY)

def sementic_retrieval():

    vectorstore=QdrantVectorStore.from_existing_collection(collection_name=CHILD_COLLECTION_NAME,embedding=embeddings,api_key=QDRANT_API_KEY,url=QDRANT_URL)

    sementic=vectorstore.as_retriever(search_kwargs={"k":5})

    return sementic

def bm25_retrieval():
    results = client.scroll(
        collection_name=CHILD_COLLECTION_NAME,
        limit=200,
        with_payload=True,
    )
    
    docs = [
        Document(
            page_content=r.payload["page_content"],
            metadata=r.payload["metadata"]
        )
        for r in results[0]
    ]
    
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 5
    return bm25

def hybrid_retrieval(sementic,bm25,query):
    hybrid=EnsembleRetriever(retrievers=[sementic,bm25],weights=[0.7,0.3])

    return hybrid.invoke(query)


def reranker(query,hybrid):
    reranker=CohereRerank(model="rerank-v3.5",top_n=5,cohere_api_key=COHERE_API_KEY)

    return reranker.compress_documents(hybrid,query)



def fetch_parents(reranked):
    parent_ids = list(set([
        doc.metadata["parent_id"] 
        for doc in reranked
    ]))
    
    results = client.scroll(
        collection_name=PARENT_COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.parent_id",
                    match=models.MatchAny(any=parent_ids)
                )
            ]
        ),
        with_payload=True,
        limit=len(parent_ids)
    )
    
    parents = [
        Document(
            page_content=r.payload["page_content"],
            metadata=r.payload["metadata"]
        )
        for r in results[0]
    ]
    
    return parents
                                      
                        