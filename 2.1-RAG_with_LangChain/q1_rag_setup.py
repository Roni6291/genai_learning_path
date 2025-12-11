import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

# Load environment variables from .env file
load_dotenv()

llm = ChatOllama(
    model='gemma3:4b',
    temperature=0.8,
)
embeddings = OllamaEmbeddings(model='llama3')
vector_store = InMemoryVectorStore(embedding=embeddings)

# loading the document
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
assert len(docs) == 1, "Expected to load exactly one document."
print(f"Total characters: {len(docs[0].page_content)}")

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(splits)} sub-documents.")

# adding the splits to the vector store
doc_ids = vector_store.add_documents(splits)
print(doc_ids[:3])

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    return SystemMessage(
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[prompt_with_context],
)

for event in agent.stream(
    {
        "messages": [HumanMessage("What is Task Decomposition?")]
    },
    stream_mode='values',
):
    event["messages"][-1].pretty_print()