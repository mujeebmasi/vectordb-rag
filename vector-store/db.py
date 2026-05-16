from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_Book"}
    ),

    Document(
        page_content="Data science involves data analysis in Python.",
        metadata={"source": "Data_Science_Book"}
    ),

    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "Deep_Learning_Book"}
    )
]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
