"""
Attention Analysis with Pre-trained Models
Task: Extract attention patterns from BERT and GPT-2 for the sentence:
"He went to the bank to deposit money."
"""

import torch
from loguru import logger
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Sentence to analyze
sentence = "He went to the bank to deposit money."

logger.info("=" * 70)
logger.info("ATTENTION ANALYSIS WITH PRE-TRAINED MODELS")
logger.info("=" * 70)
logger.info(f"\nSentence: '{sentence}'")

# ============================================================================
# PART 1: BERT Analysis
# ============================================================================

logger.info("\n" + "=" * 70)
logger.info("PART 1: BERT (bert-base-uncased)")
logger.info("=" * 70)

# Load BERT tokenizer and model
logger.info("\nLoading BERT model and tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

# Tokenize the sentence
bert_inputs = bert_tokenizer(sentence, return_tensors="pt")
logger.info(
    f"\nTokenized input: {bert_tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'][0])}"
)
logger.info(f"Number of tokens: {len(bert_inputs['input_ids'][0])}")

# Forward pass through BERT
logger.info("\nRunning forward pass through BERT...")
with torch.no_grad():
    bert_outputs = bert_model(**bert_inputs)

# Extract attention tensors
bert_attentions = bert_outputs.attentions

# Analysis
logger.info("\n--- BERT Attention Analysis ---")
logger.info(f"Number of attention layers: {len(bert_attentions)}")
logger.info(f"\nLast layer attention shape: {bert_attentions[-1].shape}")
logger.info(
    f"Shape interpretation: (batch_size, num_heads, seq_len, seq_len) = {bert_attentions[-1].shape}"
)

# Additional info
logger.info("\nBERT Configuration:")
logger.info(f"  - Total layers: {len(bert_attentions)}")
logger.info(f"  - Number of attention heads: {bert_attentions[-1].shape[1]}")
logger.info(f"  - Sequence length: {bert_attentions[-1].shape[2]}")

# ============================================================================
# PART 2: GPT-2 Analysis
# ============================================================================

logger.info("\n" + "=" * 70)
logger.info("PART 2: GPT-2 (gpt2)")
logger.info("=" * 70)

# Load GPT-2 tokenizer and model
logger.info("\nLoading GPT-2 model and tokenizer...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)

# Tokenize the sentence
gpt2_inputs = gpt2_tokenizer(sentence, return_tensors="pt")
logger.info(
    f"\nTokenized input: {gpt2_tokenizer.convert_ids_to_tokens(gpt2_inputs['input_ids'][0])}"
)
logger.info(f"Number of tokens: {len(gpt2_inputs['input_ids'][0])}")

# Forward pass through GPT-2
logger.info("\nRunning forward pass through GPT-2...")
with torch.no_grad():
    gpt2_outputs = gpt2_model(**gpt2_inputs)

# Extract attention tensors
gpt2_attentions = gpt2_outputs.attentions

# Analysis
logger.info("\n--- GPT-2 Attention Analysis ---")
logger.info(f"Number of attention layers: {len(gpt2_attentions)}")
logger.info(f"\nLast layer attention shape: {gpt2_attentions[-1].shape}")
logger.info(
    f"Shape interpretation: (batch_size, num_heads, seq_len, seq_len) = {gpt2_attentions[-1].shape}"
)

# Additional info
logger.info("\nGPT-2 Configuration:")
logger.info(f"  - Total layers: {len(gpt2_attentions)}")
logger.info(f"  - Number of attention heads: {gpt2_attentions[-1].shape[1]}")
logger.info(f"  - Sequence length: {gpt2_attentions[-1].shape[2]}")

# ============================================================================
# COMPARISON
# ============================================================================

logger.info("\n" + "=" * 70)
logger.info("COMPARISON: BERT vs GPT-2")
logger.info("=" * 70)

logger.info("\nModel Architecture Comparison:")
logger.info("  BERT:")
logger.info(f"    - Layers: {len(bert_attentions)}")
logger.info(f"    - Heads per layer: {bert_attentions[-1].shape[1]}")
logger.info(f"    - Tokens for this sentence: {bert_attentions[-1].shape[2]}")
logger.info("    - Attention type: Bidirectional (can attend to all tokens)")
logger.info("\n  GPT-2:")
logger.info(f"    - Layers: {len(gpt2_attentions)}")
logger.info(f"    - Heads per layer: {gpt2_attentions[-1].shape[1]}")
logger.info(f"    - Tokens for this sentence: {gpt2_attentions[-1].shape[2]}")
logger.info("    - Attention type: Causal (can only attend to previous tokens)")

logger.info("\nTokenization Differences:")
logger.info(f"  BERT tokens: {len(bert_inputs['input_ids'][0])}")
logger.info(f"  GPT-2 tokens: {len(gpt2_inputs['input_ids'][0])}")
logger.info("  Difference: BERT adds [CLS] and [SEP] tokens")

logger.info("\n" + "=" * 70)
logger.info("ANALYSIS COMPLETE")
logger.info("=" * 70)
