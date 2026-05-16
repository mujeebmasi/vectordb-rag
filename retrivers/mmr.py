from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

docs = [
    Document(page_content="Gradient descent is an optimization algorithm."),

    Document(page_content="Gradient descent minimizes the loss function."),

    Document(page_content="Gradient descent updates weights iteratively."),

    Document(page_content="Neural networks use gradient descent for training."),

    Document(page_content="Support Vector Machines are supervised learning models."),

    Document(page_content="Random forests are ensemble learning algorithms."),

    Document(page_content="K-Means is an unsupervised clustering algorithm."),

    Document(page_content="Linear regression predicts continuous values."),

    Document(page_content="Transformers are widely used in NLP."),

    Document(page_content="CNNs are powerful for image processing.")
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs,embeddings)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs = {"k":3}
)
print("Similarity Search Results:")
similarity_docs = similarity_retriever.invoke("What is gradient descent?")
for doc in similarity_docs:
    print(doc.page_content)
    
mmr_retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3}
    )
print("\nMMR Search Results:")
mmr_docs = mmr_retriever.invoke("What is gradient descent?")
for doc in mmr_docs:
    print(doc.page_content)