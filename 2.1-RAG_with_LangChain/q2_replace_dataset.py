import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger


"""
make sure to have the following in your .env file:

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your-langsmith\langchain-api-key>
LANGSMITH_PROJECT=rag_with_langchain
OLLAMA_API_KEY=<your-ollama-api-key>
"""
load_dotenv()

llm = ChatOllama(
    model='gemma3:4b',
    temperature=0.8,
)

# vector store setup
embeddings = OllamaEmbeddings(model='llama3')

if os.path.exists("store\\will_of_dumbledore_index"):
    logger.info("Loading existing FAISS index...")
    vector_store = FAISS.load_local("store\\will_of_dumbledore_index", embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Loaded vector store with {vector_store.index.ntotal} vectors.")
else:
    # loading the document
    loader = PyPDFLoader('data\\The_will_of_Albus_Dumbledore.pdf')
    docs = loader.load()
    logger.info(f"Total pages loaded: {len(docs)}")

    # splitting the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split PDF into {len(splits)} sub-documents.")

    logger.info("Creating new FAISS index...")
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local("store\\will_of_dumbledore_index")
    logger.info(f"Vector store created with {vector_store.index.ntotal} vectors and saved.")

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
logger.info("Created retriever from vector store.")

# Define RAG prompt template
template = """You are an assistant for question-answering tasks. 
Only use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test the RAG chain with debugging
query1 = "What does Dumbledore's will say about Harry Potter?"
logger.info(f"Query 1: {query1}")

# Check what documents are retrieved
retrieved_docs = retriever.invoke(query1)
logger.info(f"Retrieved {len(retrieved_docs)} documents")
for i, doc in enumerate(retrieved_docs):
    logger.info(f"Doc {i+1} excerpt: {doc.page_content[:200]}...")

response1 = rag_chain.invoke(query1)
logger.info(f"Response 1: {response1}")

query2 = "What items were left to Hermione Granger?"
logger.info(f"\nQuery 2: {query2}")

retrieved_docs2 = retriever.invoke(query2)
logger.info(f"Retrieved {len(retrieved_docs2)} documents")
for i, doc in enumerate(retrieved_docs2):
    logger.info(f"Doc {i+1} excerpt: {doc.page_content[:200]}...")

response2 = rag_chain.invoke(query2)
logger.info(f"Response 2: {response2}")