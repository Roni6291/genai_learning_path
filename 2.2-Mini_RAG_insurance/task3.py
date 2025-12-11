# use BM25+ to retrieve relevant chunks for user questions and
# use the TF-IDF vector store and its cosine similarity Reranker to re-rank the BM25+ results
# using sentence aware chunking strategy as it was found superior in task2.py

from pathlib import Path

from loguru import logger

from chunking.sentence_aware import sentence_aware_chunking
from retrieval.bm25 import BM25PlusVectorStore
from retrieval.tf_idf import TFIDFVectorStore

doc_path = Path("Springboard_GenAI Intermediate_Level-2_Assignment/doc.txt")

# Configure logger to write to file and terminal
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{Path(__file__).stem}.log"
logger.add(log_file, mode="w", level="INFO")

ques_path = Path("Springboard_GenAI Intermediate_Level-2_Assignment/questions.txt")
questions = []
with ques_path.open("r", encoding="utf-8") as f:
    for question in f:
        question = question.rstrip("?\n").strip()
        if question:
            questions.append(question)

logger.info(f"Total questions loaded: {len(questions)}")

# document chunking
chunks = list(sentence_aware_chunking(doc_path, max_sentences=3))
logger.info(f"Total chunks created: {len(chunks)}")

# intialize and create in memory vector stores for TF-IDF vectorization and BM25+
tfidf_vector_store = TFIDFVectorStore()
tfidf_vector_store.vectorize(chunks)

bm25_vector_store = BM25PlusVectorStore()
bm25_vector_store.vectorize(chunks)

# create BM25+ retriever
bm25retriever = bm25_vector_store.as_retriever()
tfidfreranker = tfidf_vector_store.as_reranker()

for i, question in enumerate(questions):
    logger.info("=" * 80)
    logger.info(f"Processing Question {i + 1}: {question}")
    logger.info("=" * 80)

    logger.info("Retrieving using BM25+ Retriever")
    bm25_results = bm25retriever.retrieve(
        question, top_k=6
    )  # retrieve top 6 to rerank later

    retrieved_texts = []
    logger.info(f"BM25+ retrieved {len(bm25_results)} results")
    for text, score in bm25_results:
        logger.info(f"(Score: {score:.4f}) - Text: {text}")
        retrieved_texts.append(text)

    # get retrieved texts for reranking
    results = tfidfreranker.rerank(
        query=question,
        texts=retrieved_texts,
        top_k=3,
    )
    logger.success(f"Reranked {len(results)} results using TF-IDF Reranker")
    for rank, (text, score) in enumerate(results, start=1):
        logger.info(f"Rank {rank}: (Score: {score:.4f}) - Text: {text}")

    logger.info("")  # Empty line for readability
