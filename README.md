# 📚 AskYourPDF – Conversational PDF Assistant with RAG

Turn static PDF documents into an interactive AI-powered knowledge assistant.

AskYourPDF is a Retrieval-Augmented Generation (RAG) application that enables users to upload PDF files and interact with their content through natural language conversations. Instead of manually searching through lengthy documents, users can ask questions and receive context-aware responses grounded in the uploaded document.

Built using Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Groq-powered Llama models, this project demonstrates a complete end-to-end RAG workflow.

---

## 🚀 Key Features

* 📄 Upload and analyze PDF documents instantly
* 💬 Ask questions directly from document content
* 🧠 Context-aware conversational memory for follow-up questions
* 🔍 Semantic search powered by vector embeddings
* ⚡ High-speed response generation using Groq LLMs
* 📚 FAISS-based vector storage for efficient retrieval
* 🧩 Automated document chunking and embedding generation
* 🌐 Clean and interactive Streamlit interface

---

## 🏗️ Tech Stack

| Component           | Technology              |
| ------------------- | ----------------------- |
| Frontend            | Streamlit               |
| LLM                 | Groq (Llama Models)     |
| Framework           | LangChain               |
| Embeddings          | Hugging Face Embeddings |
| Vector Database     | FAISS                   |
| Document Processing | PyPDF                   |
| Language            | Python                  |

---

## 🔄 Application Workflow

```text
PDF Upload
    ↓
Document Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Store
    ↓
Semantic Retrieval
    ↓
Groq LLM
    ↓
Context-Aware Response
```

---

## 📂 Project Structure

```text
AskYourPDF/
│
├── main.py
├── requirements.txt
├── .env
├── README.md
├── temp_uploads/
└── vectors/
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd AskYourPDF
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### 6. Run the Application

```bash
streamlit run main.py
```

---

## 💡 Example

**User:**

> What is Retrieval-Augmented Generation?

**Assistant:**

> Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. Relevant document chunks are retrieved first and then provided to the language model to generate accurate, context-aware answers grounded in the source content.

---

## 🎯 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Conversational AI
* Large Language Models (LLMs)
* Prompt Engineering
* Document Question Answering
* Streamlit Application Development
* LangChain Framework
* FAISS Vector Search

---

## 🔮 Future Enhancements

* Multi-PDF support
* Source citations in responses
* Hybrid search (Keyword + Semantic Search)
* Persistent chat history
* Cloud deployment
* RAG evaluation metrics
* User authentication

---

## 🙌 Conclusion

AskYourPDF showcases how Retrieval-Augmented Generation can transform static documents into intelligent conversational systems. By combining semantic retrieval, vector search, and large language models, the application delivers accurate, context-aware answers directly from uploaded PDFs while maintaining a natural chat experience.

If you found this project useful, feel free to ⭐ star the repository and explore the code.
