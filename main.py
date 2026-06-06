import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AskYourPDF",
    page_icon="📚",
    layout="wide",
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .title-text {
            font-size: 42px;
            font-weight: 700;
        }

        .subtitle-text {
            color: #9aa0a6;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# ENV
# ==========================================

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error(
        "GROQ_API_KEY not found. Add it to .env locally or Streamlit Secrets."
    )
    st.stop()

# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 AskYourPDF")

    st.markdown("---")

    st.markdown(
        """
        ### About

        Upload one or more PDF files and chat with them.

        **Tech Stack**
        - Groq
        - LangChain
        - FAISS
        - HuggingFace
        - Streamlit
        """
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title-text">📚 AskYourPDF</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle-text">Upload PDFs and start chatting with your documents.</div>',
    unsafe_allow_html=True,
)

st.markdown("")

# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True,
)

# ==========================================
# PROCESS PDFS
# ==========================================

if uploaded_files and not st.session_state.pdf_ready:

    with st.spinner(
        "📄 Processing PDFs, Please wait...!"
    ):

        try:

            all_docs = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(uploaded_file.read())
                    pdf_path = tmp_file.name

                loader = PyPDFLoader(pdf_path)

                docs = loader.load()

                all_docs.extend(docs)

            if len(all_docs) == 0:

                st.error(
                    "No readable content found in uploaded PDFs."
                )

                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            chunks = splitter.split_documents(
                all_docs
            )

            if len(chunks) == 0:

                st.error(
                    "Unable to create chunks from uploaded PDFs."
                )

                st.stop()

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings,
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_ready = True

            st.success(
                Print("Sucessfully Uploaded!")
            )

        except Exception as e:

            st.error(
                f"Error while processing PDFs: {str(e)}"
            )

            st.stop()

# ==========================================
# STATUS
# ==========================================

if not uploaded_files:

    st.info(
        "👆 Upload one or more PDF files to begin."
    )

elif st.session_state.pdf_ready:

    st.success(
        "✅ PDFs are ready. You can now ask questions."
    )

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

# ==========================================
# CHAT SECTION
# ==========================================

if st.session_state.pdf_ready:

    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                try:

                    docs = st.session_state.vectorstore.similarity_search(
                        question,
                        k=4
                    )

                    context = "\n\n".join(
                        [doc.page_content for doc in docs]
                    )

                    prompt = f"""
You are a helpful PDF assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, respond:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}
"""

                    response = llm.invoke(prompt)

                    answer = response.content

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Error generating response: {str(e)}"
                    )
