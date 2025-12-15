"""
Calculate ROUGE, BLEU, and Semantic Similarity scores for actual vs predicted answers
"""

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from loguru import logger

# Configure logger to write to file
logger.add("q6.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

# Sample data: Actual vs Predicted answers
data = [
    {
        "actual": "The capital of France is Paris.",
        "predicted": "Paris is the capital city of France."
    },
    {
        "actual": "The Great Wall of China is in Beijing.",
        "predicted": "The Great Wall is located in China, near Beijing."
    },
    {
        "actual": "Water boils at 100 degrees Celsius.",
        "predicted": "At 100°C, water starts to boil."
    },
    {
        "actual": "Python is a popular programming language.",
        "predicted": "Many developers use Python as a programming language."
    }
]

# Initialize scorers
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
smoothing = SmoothingFunction()

logger.info("=" * 80)
logger.info("EVALUATION METRICS: ROUGE, BLEU, and Semantic Similarity")
logger.info("=" * 80)

for i, pair in enumerate(data, 1):
    actual = pair["actual"]
    predicted = pair["predicted"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Pair {i}:")
    logger.info(f"  Actual:    {actual}")
    logger.info(f"  Predicted: {predicted}")
    logger.info(f"{'='*80}")
    
    # 1. ROUGE Scores
    rouge_scores = rouge_scorer_obj.score(actual, predicted)
    logger.info("\n1. ROUGE Scores:")
    logger.info(f"   ROUGE-1: Precision={rouge_scores['rouge1'].precision:.4f}, "
          f"Recall={rouge_scores['rouge1'].recall:.4f}, "
          f"F1={rouge_scores['rouge1'].fmeasure:.4f}")
    logger.info(f"   ROUGE-2: Precision={rouge_scores['rouge2'].precision:.4f}, "
          f"Recall={rouge_scores['rouge2'].recall:.4f}, "
          f"F1={rouge_scores['rouge2'].fmeasure:.4f}")
    logger.info(f"   ROUGE-L: Precision={rouge_scores['rougeL'].precision:.4f}, "
          f"Recall={rouge_scores['rougeL'].recall:.4f}, "
          f"F1={rouge_scores['rougeL'].fmeasure:.4f}")
    
    # 2. BLEU Score
    # Tokenize sentences
    reference = [actual.lower().split()]  # BLEU expects list of references
    candidate = predicted.lower().split()
    
    # Calculate BLEU with smoothing (to handle cases with no n-gram matches)
    bleu_score = sentence_bleu(reference, candidate, 
                               smoothing_function=smoothing.method1)
    logger.info(f"\n2. BLEU Score: {bleu_score:.4f}")
    
    # 3. Semantic Similarity (using Sentence Transformers)
    embeddings = model.encode([actual, predicted], convert_to_tensor=True)
    cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    logger.info(f"\n3. Semantic Similarity (Cosine): {cosine_similarity:.4f}")
    logger.info(f"{'='*80}\n")
