"""
GPT-2 Medium Chatbot with RAG (Retrieval-Augmented Generation)

Features:
- Loads the fine-tuned GPT-2 Medium model
- If a RAG index exists (out/rag_index/), retrieves relevant passages
  from FineWeb-Edu to ground the model's responses
- Falls back to generation-only if no RAG index is available

Usage:
    python chat_llm.py
"""

import os
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --- Configuration ---
MODEL_DIR = "out/models/gpt2_medium_finetuned"
RAG_DIR = "out/rag_index"
TOP_K = 3              # number of passages to retrieve
MAX_NEW_TOKENS = 200   # max tokens in response
TEMPERATURE = 0.7
TOP_P = 0.9

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model & Tokenizer ---
print(f"Loading model from: {MODEL_DIR}")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
model.eval()

# --- Load RAG Index (if available) ---
rag_available = False
index = None
passages = None
embedder = None

if os.path.exists(os.path.join(RAG_DIR, "faiss_index.bin")):
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        print("Loading RAG knowledge base...")
        index = faiss.read_index(os.path.join(RAG_DIR, "faiss_index.bin"))
        passages = np.load(
            os.path.join(RAG_DIR, "passages.npy"),
            allow_pickle=True
        )
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        rag_available = True
        print(f"RAG loaded: {index.ntotal:,} passages indexed")
    except ImportError:
        print("RAG dependencies not installed (faiss-cpu, sentence-transformers).")
        print("Running in generation-only mode.")
    except Exception as e:
        print(f"Failed to load RAG index: {e}")
        print("Running in generation-only mode.")
else:
    print("No RAG index found. Running in generation-only mode.")
    print("To enable RAG, run: python build_rag_index.py")


def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Retrieve relevant passages from the FAISS index."""
    if not rag_available:
        return ""

    # Embed the query
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, top_k)

    # Build context string
    context_parts = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(passages) and score > 0.2:  # relevance threshold
            context_parts.append(f"[{i+1}] {passages[idx]}")

    return "\n\n".join(context_parts)


def generate_response(prompt: str) -> str:
    """Generate a response, optionally augmented with retrieved context."""
    # Try to retrieve relevant context
    context = retrieve_context(prompt)

    if context:
        # RAG mode: prepend context
        full_prompt = (
            f"Use the following information to answer the question.\n\n"
            f"Information:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer:"
        )
    else:
        # Generation-only mode
        full_prompt = prompt

    # Tokenize (truncate if too long for model context)
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.n_positions - MAX_NEW_TOKENS
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode (skip the prompt tokens)
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()


# --- Chat Loop ---
print("\n" + "=" * 60)
print("  GPT-2 Medium Chatbot")
if rag_available:
    print("  Mode: RAG-enhanced (searching FineWeb-Edu)")
else:
    print("  Mode: Generation only")
print("  Type 'exit' to quit")
print("=" * 60 + "\n")

while True:
    prompt = input("You: ").strip()
    if not prompt:
        continue
    if prompt.lower() in ["exit", "quit"]:
        print("Exiting chatbot...")
        break

    response = generate_response(prompt)

    if rag_available:
        context = retrieve_context(prompt)
        if context:
            print(f"  [RAG: retrieved {len(context.split('['))-1} relevant passages]")

    print(f"Bot: {response}\n")
