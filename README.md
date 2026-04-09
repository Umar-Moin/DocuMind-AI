# 🧠 DocuMind AI – Intelligent Document Assistant

An AI-powered document assistant that enables users to upload PDFs and interact with them using natural language queries.  
Built using a Retrieval-Augmented Generation (RAG) pipeline to ensure accurate, context-aware responses grounded in user-provided documents.

---

## 🌟 Features

- 📄 **PDF Document Upload & Parsing**
- 🔍 **Semantic Search with FAISS**
- 🤖 **Context-Aware Question Answering (RAG)**
- ⚡ **Fast Retrieval using Vector Embeddings**
- 🌐 **Interactive Web Interface (FastAPI + HTML/CSS/JS)**
- 🧠 **LLM Integration via OpenRouter/OpenAI**

---

## 🏗️ Architecture

### Backend (FastAPI)
- Handles document upload and processing
- Implements RAG pipeline:
  - Document loading (PDF)
  - Text chunking
  - Embedding generation
  - Vector storage (FAISS)
- Processes user queries and retrieves relevant context
- Sends context to LLM for response generation

### Frontend
- Minimal, responsive UI
- File upload system
- Chat interface for Q&A
- Displays answers with optional source context

---

## 🧠 How It Works

### 1. Document Processing
- PDF is loaded and parsed
- Text is split into smaller chunks
- Each chunk is converted into vector embeddings
- Stored in FAISS vector database

### 2. Query Processing
- User submits a question
- Query is embedded into vector form
- FAISS retrieves most relevant chunks
- Retrieved context is passed to LLM
- LLM generates a grounded response

---

## 🛠️ Tech Stack

### Core
- **Python**
- **FastAPI**
- **LangChain**

### AI / ML
- **OpenRouter / OpenAI (LLM & Embeddings)**
- **FAISS (Vector Database)**

### Frontend
- **HTML, CSS, JavaScript**

---

## 🚀 Installation
```bash
git clone https://github.com/your-username/documind-ai.git
cd documind-ai
```
## 2. Create virtual environment
````bash
python -m venv venv
venv\Scripts\activate   # Windows
````
## 3. Install dependencies
````bash
pip install -r requirements.txt
````
## 4. Setup environment variables

Create a .env file:
````bash
OPENAI_API_KEY=your_api_key_here
````

## ▶️ Running the Application

Start the backend server:
````bash
uvicorn api:app --reload
````
Open index.html in your browser to use the application.

## 🎯 Usage
- Upload a PDF document
- Click Process Files
- Ask questions about the document
- Receive context-aware answers
## ⚠️ Limitations
- Answers are limited to uploaded documents (pure RAG system)
- No external knowledge or web search integration
- Performance depends on document structure and chunking strategy
## 🔮 Future Improvements
- Agentic AI (RAG + Web Search integration)
- Multi-document support with memory
- Cloud deployment with persistent storage
- Authentication & user sessions
## 🧠 Key Learnings
- Built an end-to-end RAG pipeline
- Implemented vector search with FAISS
- Integrated LLMs with external data sources
- Developed a full-stack AI application
- Understood limitations of LLM grounding vs general knowledge
## 📄 License

This project is for educational and demonstration purposes.

