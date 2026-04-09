from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import glob
import os

load_dotenv()


def load_documents(data_folder="data/"):
    all_docs = []
    pdf_files = glob.glob(f"{data_folder}*.pdf")

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{data_folder}'")

    for path in pdf_files:
        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)

    return all_docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )


def load_or_create_vectorstore(chunks=None, force_rebuild=False):
    embeddings = get_embeddings()

    if os.path.exists("vectorstore") and not force_rebuild:
        print("Loading existing vectorstore...")
        return FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

    if chunks is None:
        raise ValueError("No vectorstore found and no chunks provided to create one.")

    print("Creating new vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    return vectorstore


def ask(vectorstore, llm, question, chat_history):
    try:
        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            return "Nothing relevant found in the documents.", chat_history, []

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
        messages.append(HumanMessage(content=question))

        response = llm.invoke(messages)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response.content))

        return response.content, chat_history, sources

    except Exception as e:
        return f"Something went wrong: {str(e)}", chat_history, []


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages across {len(set(d.metadata['source'] for d in docs))} file(s)")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = load_or_create_vectorstore(chunks)
    print("Vectorstore ready\n")

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    chat_history = []

    while True:
        question = input("Question (or 'exit'): ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            break

        answer, chat_history, sources = ask(vectorstore, llm, question, chat_history)

        print(f"\nAnswer: {answer}")
        print(f"Sources: {', '.join(sources)}\n")