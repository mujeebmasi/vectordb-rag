from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model =HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
    
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":20,
                     "fetch_k":10,  #fetch_k is the number of documents to fetch before applying MMR
                                    #here, before applying mmr, we must have atleast 10 documents to choose from, so fetch_k should be greater than k
                     "lambda_mult":0.5 #lamda = 0 means highly diverse, 1 means less diverse.
                     }
                     )

llm = ChatMistralAI(
    model="mistral-small"
)
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

print("RAG SYSTEM CREATED")
print("press 0 to exit")

while True:
    query = input("\nEnter your question: ")
    if query == "0":
        print("Exiting the RAG system. Goodbye!")
        break
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
        )
    final_prompt = prompt.format_messages(
    context=context,
    question=query
)
    
    response = llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")
