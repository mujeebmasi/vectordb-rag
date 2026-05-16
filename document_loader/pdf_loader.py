from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import data
data = PyPDFLoader("document loader/PRD_Full_Stack_Training.pdf")
docs = data.load()
print(docs[3].page_content) 
