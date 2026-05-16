import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")


# -----------------------------------
# Upload PDF
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF book",
    type="pdf"
)


# -----------------------------------
# Create Vector Database
# -----------------------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            # Load PDF
            loader = PyPDFLoader(file_path)

            docs = loader.load()

            # Chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            # HuggingFace Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Create Chroma DB
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

        st.success("Vector database created successfully 🚀")


# -----------------------------------
# Load Existing Vector DB
# -----------------------------------

if os.path.exists("chroma_db"):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # LLM
    llm = ChatMistralAI(
        model="mistral-small"
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),

            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()

    st.subheader("Ask Questions From the PDF")

    query = st.text_input("Enter your question")

    if query:

        with st.spinner("Searching document..."):

            # Retrieve Relevant Docs
            docs = retriever.invoke(query)

            # Context Creation
            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            # Final Prompt
            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            # LLM Response
            response = llm.invoke(final_prompt)

            # Output
            st.write("### 🤖 AI Answer")

            st.write(response.content)

            # Debug Retrieved Chunks
            with st.expander("📄 Retrieved Chunks"):

                for i, doc in enumerate(docs):

                    st.write(f"Chunk {i+1}")

                    st.write(doc.page_content)

                    st.divider()