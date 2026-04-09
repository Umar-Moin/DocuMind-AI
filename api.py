from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import shutil

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vectorstore = None
llm = None
chat_history = []


def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )


def get_llm():
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )


@app.on_event("startup")
def startup():
    global vectorstore, llm
    llm = get_llm()
    if os.path.exists("vectorstore"):
        print("Loading existing vectorstore on startup...")
        vectorstore = FAISS.load_local(
            "vectorstore",
            get_embeddings(),
            allow_dangerous_deserialization=True
        )


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global vectorstore

    os.makedirs("data", exist_ok=True)
    all_chunks = []

    for file in files:
        path = f"data/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file.filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    vectorstore = FAISS.from_documents(all_chunks, get_embeddings())
    vectorstore.save_local("vectorstore")

    return {"message": f"Processed {len(files)} file(s), {len(all_chunks)} chunks created."}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(body: QuestionRequest):
    global vectorstore, llm, chat_history

    if vectorstore is None:
        return {"error": "No documents loaded. Please upload a PDF first."}

    try:
        docs = vectorstore.similarity_search(body.question, k=3)

        if not docs:
            return {"answer": "Nothing relevant found in the documents.", "sources": []}

        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))

        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Answer using only the context provided below. "
                "If the answer is not in the context, say: "
                "'I could not find an answer based on the provided documents.'\n\n"
                f"Context:\n{context}"
            ))
        ]

        messages.extend(chat_history)
        messages.append(HumanMessage(content=body.question))

        response = llm.invoke(messages)

        chat_history.append(HumanMessage(content=body.question))
        chat_history.append(AIMessage(content=response.content))

        return {"answer": response.content, "sources": sources}

    except Exception as e:
        return {"error": str(e)}


@app.post("/reset")
def reset_chat():
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared."}