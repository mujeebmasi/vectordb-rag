from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

docs = PyPDFLoader("D:\AI\RagLearning\document_loader\PRD_Full_Stack_Training.pdf").load()

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)