import numpy as np
from loguru import logger


def calculate_attention(query, key, value):
    """
    Calculate attention weights and output using the scaled dot-product attention mechanism.

    Parameters:
    -----------
    query : numpy.ndarray
        Query vectors of shape (seq_len_q, d_k) or (batch_size, seq_len_q, d_k)
    key : numpy.ndarray
        Key vectors of shape (seq_len_k, d_k) or (batch_size, seq_len_k, d_k)
    value : numpy.ndarray
        Value vectors of shape (seq_len_v, d_v) or (batch_size, seq_len_v, d_v)
        Note: seq_len_k must equal seq_len_v

    Returns:
    --------
    output : numpy.ndarray
        Attention output of shape (seq_len_q, d_v) or (batch_size, seq_len_q, d_v)
    attention_weights : numpy.ndarray
        Attention weights of shape (seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k)

    Formula:
    --------
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    # Get the dimension of the key vectors (d_k)
    d_k = key.shape[-1]

    # Calculate attention scores: Q * K^T
    # Using matmul for proper handling of both 2D and 3D arrays
    scores = np.matmul(query, np.swapaxes(key, -1, -2))

    # Scale by sqrt(d_k)
    scaled_scores = scores / np.sqrt(d_k)

    # Apply softmax to get attention weights
    # Subtract max for numerical stability
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Calculate weighted sum of values
    output = np.matmul(attention_weights, value)

    return output, attention_weights


# Example usage
if __name__ == "__main__":
    # Example 1: Simple 2D case
    logger.info("Example 1: Simple attention calculation")
    logger.info("-" * 50)

    # Define query, key, and value vectors
    # Sequence length = 2, dimension = 2
    query = np.array([[1.0, 0.0]])

    key = np.array([[1.0, 1.0], [0.0, 1.0]])

    value = np.array([[1.0, 0.0], [0.0, 1.0]])

    output, attention_weights = calculate_attention(query, key, value)

    logger.info(f"Query shape: {query.shape}")
    logger.info(f"Key shape: {key.shape}")
    logger.info(f"Value shape: {value.shape}")
    logger.info(f"\nAttention weights shape: {attention_weights.shape}")
    logger.info(f"Attention weights:\n{attention_weights}")
    logger.info(f"\nOutput shape: {output.shape}")
    logger.info(f"Output:\n{output}")

    # Example 2: Batched case
    logger.info("\n\nExample 2: Batched attention calculation")
    logger.info("-" * 50)

    # Batch size = 2, sequence length = 3, dimension = 4
    query_batch = np.random.randn(2, 3, 4)
    key_batch = np.random.randn(2, 3, 4)
    value_batch = np.random.randn(2, 3, 2)

    output_batch, attention_weights_batch = calculate_attention(
        query_batch, key_batch, value_batch
    )

    logger.info(f"Query batch shape: {query_batch.shape}")
    logger.info(f"Key batch shape: {key_batch.shape}")
    logger.info(f"Value batch shape: {value_batch.shape}")
    logger.info(f"\nAttention weights batch shape: {attention_weights_batch.shape}")
    logger.info(f"Output batch shape: {output_batch.shape}")
