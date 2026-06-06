import streamlit as st
import os
import tempfile
import sys

st.write("Python Version:", sys.version)
st.stop()

from dotenv import load_dotenv

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)

from langchain_community.document_loaders import (
    PyPDFLoader,
)

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_groq import ChatGroq

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

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

        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
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
        "GROQ_API_KEY not found. Please configure your .env file or Streamlit Secrets."
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
    model_name="all-MiniLM-L6-v2",
)

# ==========================================
# CHAT HISTORY
# ==========================================

store = {}

def get_session_history(session_id):

    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

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

        Upload one or more PDF documents and chat with them using AI.

        **Powered By**
        - Groq
        - LangChain
        - FAISS
        - HuggingFace Embeddings
        """
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        store.clear()

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
    '<div class="subtitle-text">Upload PDFs, wait for indexing, and start chatting with your documents.</div>',
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
# PROCESS PDFs
# ==========================================

if uploaded_files and not st.session_state.pdf_ready:

    with st.spinner(
        "📄 Processing PDFs, Please wait."
    ):

        all_docs = []

        try:

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
                    "Unable to create chunks from PDFs."
                )

                st.stop()

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings,
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_ready = True

            st.success(
                print(" Succesfully uploaded!")
            )

        except Exception as e:

            st.error(
                f"Error while processing PDFs: {str(e)}"
            )

            st.stop()

# ==========================================
# PDF STATUS
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

session_id = "default"

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

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    retriever = (
                        st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 4}
                        )
                    )

                    contextualize_q_prompt = (
                        ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    """
                                    Given chat history and latest user question,
                                    rewrite it into a standalone question.
                                    """
                                ),
                                MessagesPlaceholder(
                                    "chat_history"
                                ),
                                (
                                    "human",
                                    "{input}"
                                ),
                            ]
                        )
                    )

                    history_aware_retriever = (
                        create_history_aware_retriever(
                            llm,
                            retriever,
                            contextualize_q_prompt,
                        )
                    )

                    qa_prompt = (
                        ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    """
                                    You are a helpful PDF assistant.

                                    Answer ONLY using the provided context.

                                    If the answer is not available in the context,
                                    say:

                                    "I couldn't find that information in the uploaded documents."

                                    Context:

                                    {context}
                                    """
                                ),
                                MessagesPlaceholder(
                                    "chat_history"
                                ),
                                (
                                    "human",
                                    "{input}"
                                ),
                            ]
                        )
                    )

                    document_chain = (
                        create_stuff_documents_chain(
                            llm,
                            qa_prompt,
                        )
                    )

                    rag_chain = (
                        create_retrieval_chain(
                            history_aware_retriever,
                            document_chain,
                        )
                    )

                    conversational_chain = (
                        RunnableWithMessageHistory(
                            rag_chain,
                            lambda sid: get_session_history(
                                sid
                            ),
                            input_messages_key="input",
                            history_messages_key="chat_history",
                            output_messages_key="answer",
                        )
                    )

                    response = conversational_chain.invoke(
                        {"input": question},
                        config={
                            "configurable": {
                                "session_id": session_id
                            }
                        }
                    )

                    answer = response["answer"]

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