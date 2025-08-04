# main.py (Final Optimized Version)

import os
import requests
import tempfile
from typing import List

# --- Web Server Imports ---
from fastapi import FastAPI, Header, Depends, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- LangChain Imports (Optimized) ---
from langchain_community.document_loaders import PyMuPDFLoader  # <-- FASTER
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- FASTER (LOCAL)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# --- API Keys ---
# <-- PASTE YOUR GROQ KEY
EXPECTED_BEARER_TOKEN = "67a31b16a70c71e4b6f37b9fb2a6e7ea81dc2da74219a281fdaac5935e076f07"  # <-- PASTE YOUR TEAM TOKEN

# --- Global Cache & Models ---
retriever_cache = {}
# Initialize the embedding model once on startup to avoid reloading it.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ======================================================================================
# 1. SERVER LIFESPAN & CACHE HELPER
# ======================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server is starting up. Initializing empty cache. ---")
    yield
    retriever_cache.clear()
    print("--- Server shutdown: Cache cleared. ---")

def get_or_create_retriever(doc_url: str):
    if doc_url in retriever_cache:
        print(f"--- Cache HIT for document: {doc_url[:50]}... ---")
        return retriever_cache[doc_url]

    print(f"--- Cache MISS for document: {doc_url[:50]}... ---")
    print("--- Building new retriever. This will be slow on first run... ---")

    try:
        response = requests.get(doc_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        loader = PyMuPDFLoader(temp_file_path) # <-- FASTER LOADER
        docs = loader.load()
        os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs)

        # Use the pre-loaded local embedding model
        vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embedding_model)

        new_retriever = vectorstore.as_retriever(search_kwargs={'k': 7}) # Using k=7 for more context
        retriever_cache[doc_url] = new_retriever
        print("--- New retriever built and cached successfully. ---")
        return new_retriever
    except Exception as e:
        print(f"FATAL: Could not create retriever for {doc_url}: {e}")
        return None

# ======================================================================================
# 2. API SETUP, MODELS, AUTH, and RAG LOGIC
# ======================================================================================

app = FastAPI(title="Hackathon Submission API", lifespan=lifespan)

class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
    token = authorization.split(" ")[1]
    if token != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return token

async def get_answers(questions: List[str], doc_url: str) -> List[str]:
    retriever = get_or_create_retriever(doc_url)
    if not retriever:
        raise RuntimeError(f"Retriever could not be created for URL: {doc_url}")

    llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192") # Using the more powerful model

    prompt = PromptTemplate(
        template="You are an expert Q&A assistant for policy and legal documents. Answer the following question based *only* on the provided context. Be concise and precise. If the answer cannot be found in the context, state that clearly. CONTEXT: {context} QUESTION: {question} ANSWER:",
        input_variables=["context", "question"],
    )
    rag_chain = prompt | llm | StrOutputParser()

    answers = []
    for q in questions:
        retrieved_docs = retriever.invoke(q)
        context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = rag_chain.invoke({"context": context_str, "question": q})
        answers.append(answer)

    return answers

# ======================================================================================
# 3. API ENDPOINT
# ======================================================================================

@app.post("/api/v1/hackrx/run", response_model=HackathonResponse)
async def run_submission(request: HackathonRequest, token: str = Depends(verify_token)):
    print(f"Received valid request for document: {request.documents[:50]}...")
    try:
        answers = await get_answers(request.questions, request.documents)
        return {"answers": answers}
    except Exception as e:
        print(f"An error occurred during RAG processing: {e}")
        # Matching the simplified response format even on error
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
