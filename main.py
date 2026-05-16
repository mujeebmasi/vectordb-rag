from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
docs = PyPDFLoader("document_loader/deeplearning.pdf").load()
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions based on the following context"),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(model="mistral-small")
prompt = template.format_messages(data=docs[2].page_content) 
result = model.invoke(prompt)
print(result.content)