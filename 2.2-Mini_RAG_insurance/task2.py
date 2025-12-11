from functools import partial
from pathlib import Path

from loguru import logger

from chunking.fixed_window import fixed_window_chunking
from chunking.sentence_aware import sentence_aware_chunking
from retrieval.bm25 import BM25PlusVectorStore
from retrieval.tf_idf import TFIDFVectorStore

# TODO: select chunking strategy
chunking_strategy = "fixed_window"  # Options: "fixed_window" or "sentence_aware"
doc_path = Path("Springboard_GenAI Intermediate_Level-2_Assignment/doc.txt")

# Configure logger to write to file and terminal
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{Path(__file__).stem}_{chunking_strategy}.log"
logger.add(log_file, mode="w", level="INFO")

chunking_method = (
    partial(sentence_aware_chunking, doc_path, max_sentences=3)
    if chunking_strategy == "sentence_aware"
    else partial(fixed_window_chunking, doc_path, chunk_size=40, overlap=10)
)

ques_path = Path("Springboard_GenAI Intermediate_Level-2_Assignment/questions.txt")
questions = []
with ques_path.open("r", encoding="utf-8") as f:
    for question in f:
        question = question.rstrip("?\n").strip()
        if question:
            questions.append(question)

logger.info(f"Total questions loaded: {len(questions)}")

# document chunking
chunks = list(chunking_method())

logger.info(f"Total chunks created: {len(chunks)}")

# intialize and create in memory vector stores for TF-IDF vectorization and BM25+
tfidf_vector_store = TFIDFVectorStore()
tfidf_vector_store.vectorize(chunks)

bm25_vector_store = BM25PlusVectorStore()
bm25_vector_store.vectorize(chunks)

# create retrievers from vector stores
tfidfretriever = tfidf_vector_store.as_retriever()
bm25retriever = bm25_vector_store.as_retriever()


for i, question in enumerate(questions):
    logger.info("=" * 80)
    logger.info(f"Processing Question {i + 1}: {question}")
    logger.info("=" * 80)

    logger.info("Retrieving using TFIDF Retriever")
    tfidf_results = tfidfretriever.retrieve(question, top_k=3)

    for text, score in tfidf_results:
        logger.info(f"Text: {text} | Similarity Score: {score:.4f}")

    logger.info("Retrieving using BM25+ Retriever")
    bm25_results = bm25retriever.retrieve(question, top_k=3)

    for text, score in bm25_results:
        logger.info(f"Text: {text} | Similarity Score: {score:.4f}")

    logger.info("")  # Empty line for readability
