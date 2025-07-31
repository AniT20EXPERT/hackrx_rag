from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import numpy as np
import nltk
import os
from nltk.tokenize import sent_tokenize

# Try to load the model but handle failures gracefully
try:
    from sentence_transformers import SentenceTransformer
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    embedding_model = None

def split_text(text: str) -> List[str]:
    """
    Use NLTK's punkt tokenizer for better sentence detection.
    """
    max_chunk_size = 700

    # Set NLTK data path
    nltk_data_paths = [
        '/opt/render/project/src/nltk_data',
        os.path.expanduser('~/nltk_data'),
        '/usr/share/nltk_data',
        '/usr/local/share/nltk_data'
    ]
    
    for path in nltk_data_paths:
        if os.path.exists(path):
            nltk.data.path.append(path)
    
    # Try to use punkt tokenizer
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        print("NLTK punkt not found, falling back to simple split")
        # Fallback to simple sentence splitting
        sentences = text.replace('! ', '!|').replace('. ', '.|').replace('? ', '?|').split('|')

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_embeddings_batch(chunks: List[str]) -> List[Tuple[str, List[float]]]:
    """
    Batch version - processes all chunks at once for maximum efficiency.
    """
    if not chunks:
        return []
    
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded")

    # Process all chunks in one call (very efficient)
    embeddings_array = embedding_model.encode(
        chunks,
        batch_size=32,  # Process 32 chunks at a time
        show_progress_bar=False,  # Disable progress bar for deployment
        convert_to_tensor=False  # Return as numpy arrays
    )

    # Return
    return [(chunk, embedding.tolist()) for chunk, embedding in zip(chunks, embeddings_array)]

def get_question_embedding(question: str) -> List[float]:
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded")
    return embedding_model.encode(question).tolist()
