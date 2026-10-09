from  config import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid 
import io
import easyocr
import numpy as np
import fitz
from qdrant_client import QdrantClient,models
from langchain_core.documents import Document
import cv2


reader=easyocr.Reader(['en'])
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client=QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY)


def is_scanned_page(page,text_threshold=50):
    text=page.get_text().strip()
    text_length=len(text)
    is_scanned=text_length<text_threshold

    return is_scanned



def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_num, page in enumerate(doc):
        if is_scanned_page(page):
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            text = ' '.join(reader.readtext(img_cv, detail=0))
            source = "ocr"
        else:
            text = page.get_text()
            source = "native"

        results.append({
            "page_num": page_num + 1,
            "text": text.strip(),
            "source": source
        })

    doc.close()
    return results

def load_chunk(path):
    page_results = extract_text_with_ocr(path)

    docs = [
        Document(
            page_content=p["text"],
            metadata={"source": path, "page": p["page_num"], "ocr_source": p["source"]}
        )
        for p in page_results
        if p["text"]
    ]

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    parent_chunks = parent_splitter.split_documents(docs)
    child_chunks = []

    for parent in parent_chunks:
        parent_id=str(uuid.uuid4())
        parent.metadata["parent_id"] = parent_id
        child_docs = child_splitter.split_documents([parent])
        for child in child_docs:
            child.metadata["parent_id"] = parent_id
            child.metadata["source"] = path

            child_chunks.extend(child_docs)

            return parent_chunks, child_chunks


def reset_collection(name):
     if client.collection_exists(name):
          client.delete_collection(name)

def store(parent_chunks,child_chunks):
     reset_collection(PARENT_COLLECTION_NAME)
     reset_collection(CHILD_COLLECTION_NAME)

     parent_vectorstore=QdrantVectorStore.from_documents(parent_chunks,embeddings,api_key=QDRANT_API_KEY,url=QDRANT_URL,collection_name=PARENT_COLLECTION_NAME)

     client.create_payload_index(collection_name=PARENT_COLLECTION_NAME,field_name="metadata.parent_id",field_schema=models.PayloadSchemaType.KEYWORD)

     child_vectorstore=QdrantVectorStore.from_documents(child_chunks,embeddings,api_key=QDRANT_API_KEY,url=QDRANT_URL,collection_name=CHILD_COLLECTION_NAME)


     return parent_vectorstore,child_vectorstore





