from pathlib import Path
from typing import Generator
from loguru import logger


def tokenizing(doc_path: Path) -> list[str]:
    with doc_path.open('r', encoding='utf-8') as f:
        content = f.read()

    # Remove all newlines, tabs, and periods, then split into list of strings
    # First, replace all newlines and tabs with spaces
    cleaned_content = content.replace('\n', ' ').replace('\t', ' ').replace('.', '')

    # Split into list of words (split() handles multiple spaces automatically)
    return cleaned_content.split()


def fixed_window_chunking(doc_path: Path, chunk_size:int=40, overlap:int=10) -> Generator[str, None, None]:
    
    doc = tokenizing(doc_path)
    
    start = 0
    doc_length = len(doc)

    while start < doc_length:
        end = min(start + chunk_size, doc_length)
        chunk = doc[start:end]
        start += chunk_size - overlap  # Move start forward by chunk_size - overlap
        yield " ".join(chunk)


if __name__ == "__main__":
    insurance_doc = Path(__file__).parent.parent / 'Springboard_GenAI Intermediate_Level-2_Assignment/doc.txt'

    chunks = []
    for chunk in fixed_window_chunking(insurance_doc):
        chunks.append(chunk)
    
    logger.info(f"Total chunks created: {len(chunks)}")
    logger.info(f"First chunk: {chunks[0]}")
    logger.info(f"Second chunk: {chunks[1]}")
    logger.info(f"Last chunk: {chunks[-1]}")