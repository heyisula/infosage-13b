"""
Build RAG Knowledge Base from FineWeb-Edu Dataset

Standalone script to build/rebuild the FAISS vector index for the RAG chatbot.
This indexes passages from FineWeb-Edu so the chatbot can retrieve relevant
context when answering questions.

Usage:
    pip install faiss-cpu sentence-transformers datasets tqdm
    python build_rag_index.py
"""

import os
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Run: pip install faiss-cpu")
    exit(1)

# --- Configuration ---
NUM_SAMPLES = 100_000       # Number of dataset rows to use
MAX_PASSAGE_LEN = 500       # Characters per chunk
MIN_PASSAGE_LEN = 50        # Skip chunks shorter than this
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256
RAG_DIR = "out/rag_index"


def main():
    os.makedirs(RAG_DIR, exist_ok=True)

    # 1. Load dataset
    print(f"Loading {NUM_SAMPLES:,} samples from FineWeb-Edu...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True
    )

    subset = dataset.take(NUM_SAMPLES)
    data_list = [row for row in tqdm(subset, total=NUM_SAMPLES, desc="Downloading")]

    # 2. Chunk into passages
    print("Chunking into passages...")
    passages = []
    for row in tqdm(data_list, desc="Chunking"):
        text = row["text"].strip()
        for i in range(0, len(text), MAX_PASSAGE_LEN):
            chunk = text[i:i + MAX_PASSAGE_LEN].strip()
            if len(chunk) >= MIN_PASSAGE_LEN:
                passages.append(chunk)

    print(f"Total passages: {len(passages):,}")

    # 3. Embed
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Encoding passages...")
    embeddings = embedder.encode(
        passages,
        show_progress_bar=True,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True
    )

    # 4. Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 5. Save
    faiss.write_index(index, os.path.join(RAG_DIR, "faiss_index.bin"))
    np.save(os.path.join(RAG_DIR, "passages.npy"), np.array(passages, dtype=object))

    print(f"\nRAG index saved to {RAG_DIR}/")
    print(f"  Vectors: {index.ntotal:,} ({dimension}D)")
    print(f"  Passages: {len(passages):,}")
    print("Done!")


if __name__ == "__main__":
    main()
