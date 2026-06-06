import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

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

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)

from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import (
    PyPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# ENV
# -------------------------

load_dotenv()

# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# -------------------------
# CHAT HISTORY
# -------------------------

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

# -------------------------
# STREAMLIT UI
# -------------------------

st.title("📄 AskYourPDF")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

session_id = "default"

# -------------------------
# PROCESS PDF
# -------------------------

if uploaded_file:

    if "vectorstore" not in st.session_state:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            st.error(
        "No readable text found in the uploaded PDF."
        )
    st.stop()

    vectorstore = FAISS.from_documents(
         chunks,
         embeddings
         )

    st.session_state.vectorstore = vectorstore

    st.success("PDF processed successfully!")

# -------------------------
# CHAT
# -------------------------

if "vectorstore" in st.session_state:

    question = st.chat_input(
        "Ask something about the PDF..."
    )

    if question:

        retriever = (
            st.session_state.vectorstore.as_retriever()
        )

        contextualize_q_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Given chat history and latest question, create standalone question."
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
        )

        history_aware_retriever = (
            create_history_aware_retriever(
                llm,
                retriever,
                contextualize_q_prompt
            )
        )

        qa_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        Use context to answer briefly.

                        {context}
                        """
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
        )

        document_chain = (
            create_stuff_documents_chain(
                llm,
                qa_prompt
            )
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )

        conversational_chain = (
            RunnableWithMessageHistory(
                rag_chain,
                lambda sid: get_session_history(sid),
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

        st.chat_message("user").write(question)

        st.chat_message("assistant").write(
            response["answer"]
        )