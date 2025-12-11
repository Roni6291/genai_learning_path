from pathlib import Path
from typing import Generator
from loguru import logger

def text_to_sentences(doc_path: Path) -> Generator[str, None, None]:
    with doc_path.open('r', encoding='utf-8') as f:
        for line in f:
            if new_line := line.replace('\n', ' ').replace('\t', ' ').strip():
                for sentence in new_line.split('.'):
                    if new_sentence := sentence.strip():
                        yield new_sentence

def sentence_aware_chunking(doc: Path, max_sentences: int = 3) -> Generator[str, None, None]:
    sentence_buffer = []
    for sentence in text_to_sentences(doc):
        sentence_buffer.append(sentence)
        if len(sentence_buffer) == max_sentences:
            yield ". ".join(sentence_buffer)
            sentence_buffer = []
    if sentence_buffer:
        yield ". ".join(sentence_buffer)

if __name__ == "__main__":
    insurance_doc = Path(__file__).parent.parent / 'Springboard_GenAI Intermediate_Level-2_Assignment/doc.txt'
    chunks = []
    for chunk in sentence_aware_chunking(insurance_doc, max_sentences=3):
        chunks.append(chunk)
    logger.debug(f"First chunk: {chunks[0]}")
    logger.debug(f"Last chunk: {chunks[-1]}")
    logger.info(f"Total chunks created: {len(chunks)}")