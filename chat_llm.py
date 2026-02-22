"""
Llama-2-13B QLoRA Chatbot with RAG + Live HuggingFace Search

Retrieval strategy (layered):
  1. Search LOCAL FAISS index (instant, offline)
  2. If local results aren't good enough, LIVE SEARCH FineWeb-Edu on
     HuggingFace by streaming the dataset and keyword-filtering (online)
  3. Rank all retrieved passages by embedding similarity
  4. Use top passages as context for generation

Usage:
    python chat_llm.py
"""

import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)
import re
import json
import inspect
import multiprocessing
import wordsegment
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel
try:
    import faiss
except ImportError:
    faiss = None

# Configuration
BASE_MODEL_ID = "NousResearch/Llama-2-13b-hf"
ADAPTER_DIR = "out/final_model"
RAG_DIR = "out/rag_index"

TOP_K = 3                # passages to use as context
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.5
TOP_P = 0.9
REPETITION_PENALTY = 1.2

# Live search settings
LIVE_SEARCH_ENABLED = True       # set False to disable internet search
LIVE_SEARCH_SAMPLES = 10_000     # how many dataset rows to scan per query
LIVE_SEARCH_MAX_MATCHES = 50    # max keyword matches to collect
LIVE_SEARCH_MIN_SCORE = 0.2     # minimum similarity to include a passage

# Safe defaults in case functions are called before __main__ initializes
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None
embedder = None
faiss_index = None
local_passages = None
local_rag_available = False
live_search_available = False

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model & Tokenizer
    print(f"Loading base model: {BASE_MODEL_ID}")

    # Configure 4-bit quantization for local VRAM efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        clean_up_tokenization_spaces=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def patch_adapter_config(adapter_dir: str):
        """Remove unknown LoraConfig keys caused by PEFT version mismatch."""
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        from peft import LoraConfig
        valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys()) - {"self"}
        unknown_keys = [k for k in config if k not in valid_keys]

        if not unknown_keys:
            print("[patch] adapter_config.json is compatible.")
            return

        print(f"[patch] Removing unknown keys: {unknown_keys}")

        # Backup original first
        backup_path = config_path + ".bak"
        if not os.path.exists(backup_path):
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

        for k in unknown_keys:
            del config[k]

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print("[patch] adapter_config.json patched.")

    # Patch before loading adapters
    patch_adapter_config(ADAPTER_DIR)

    # Load base model
    # GPU-only loading with precise headroom
    max_memory = {
        0: "7500MiB",   # 7.3 GB, leaves 500 MB headroom for activations
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )

    # Load LoRA adapters
    print(f"Applying LoRA adapters from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    #Load Embedding Model
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedder.num_proc = 1
        print("Embedding model loaded")
    except ImportError:
        print("sentence-transformers not installed. RAG features disabled.")
        print("Install with: pip install sentence-transformers")

    # Load Local FAISS Index (if available)
    local_rag_available = False
    faiss_index = None
    local_passages = None

    if embedder:
        try:
            import faiss
            if os.path.exists(os.path.join(RAG_DIR, "faiss_index.bin")):
                print("Loading local RAG knowledge base...")
                faiss_index = faiss.read_index(os.path.join(RAG_DIR, "faiss_index.bin"))
                local_passages = np.load(
                    os.path.join(RAG_DIR, "passages.npy"),
                    allow_pickle=True
                )
                local_rag_available = True
                print(f"Local RAG loaded: {faiss_index.ntotal:,} passages indexed")
            else:
                print("No local RAG index found.")
        except ImportError:
            print("faiss-cpu not installed. Local RAG disabled.")
            print("Install with: pip install faiss-cpu")
        except Exception as e:
            print(f"Failed to load local RAG index: {e}")

    # Check Live Search Availability
    live_search_available = False
    if LIVE_SEARCH_ENABLED and embedder:
        try:
            from datasets import load_dataset
            live_search_available = True
            print("Live HuggingFace search: ENABLED")
        except ImportError:
            print("datasets library not installed. Live search disabled.")
            print("Install with: pip install datasets")
    else:
        if not LIVE_SEARCH_ENABLED:
            print("Live HuggingFace search: DISABLED (set LIVE_SEARCH_ENABLED=True to enable)")


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query for filtering."""
    stop_words = {
        "what", "is", "a", "an", "the", "how", "does", "do", "can", "could",
        "would", "should", "will", "are", "was", "were", "been", "being",
        "have", "has", "had", "having", "why", "when", "where", "which",
        "who", "whom", "this", "that", "these", "those", "am", "be", "it",
        "its", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "don",
        "now", "also", "me", "my", "tell", "explain", "describe", "define",
        "please", "i", "you", "we", "they", "he", "she",
        "hi", "hey", "hello", "thanks", "thank", "ok", "okay", "yes", "no",
        "sure", "great", "good", "bye", "goodbye"
    }

    words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
    keywords = [w for w in words if w not in stop_words]

    if not keywords:
        words_sorted = sorted(words, key=len, reverse=True)
        keywords = words_sorted[:3]

    return keywords


def search_local_index(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Search the local FAISS index. Returns list of (passage, score)."""
    if not local_rag_available or not embedder:
        return []

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_index.search(query_embedding, top_k * 2)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(local_passages) and score > 0.2:
            results.append((str(local_passages[idx]), float(score)))

    return results[:top_k]


def search_huggingface_live(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """
    Live search FineWeb-Edu on HuggingFace by streaming the dataset,
    filtering by keywords, and ranking by embedding similarity.
    """
    if not live_search_available or not embedder:
        return []

    keywords = extract_keywords(query)
    if not keywords:
        return []

    print(f"  [Live search: scanning FineWeb-Edu for '{' '.join(keywords)}'...]")

    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True
        )

        matched_passages = []
        scanned = 0

        for row in dataset:
            scanned += 1
            if scanned > LIVE_SEARCH_SAMPLES:
                break

            text = row.get("text", "").strip()
            if not text:
                continue

            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                for i in range(0, min(len(text), 2000), 500):
                    chunk = text[i:i + 500].strip()
                    if len(chunk) > 50 and any(kw in chunk.lower() for kw in keywords):
                        matched_passages.append(chunk)

                        if len(matched_passages) >= LIVE_SEARCH_MAX_MATCHES:
                            break

            if len(matched_passages) >= LIVE_SEARCH_MAX_MATCHES:
                break

        if not matched_passages:
            print(f"  [Live search: no matches in {scanned:,} samples]")
            return []

        print(f"  [Live search: found {len(matched_passages)} passages in {scanned:,} samples]")

        query_embedding = embedder.encode([query], convert_to_numpy=True)
        passage_embeddings = embedder.encode(matched_passages, convert_to_numpy=True)

        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        passage_norms = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        similarities = (query_norm @ passage_norms.T)[0]

        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score >= LIVE_SEARCH_MIN_SCORE:
                results.append((matched_passages[idx], score))

        return results

    except Exception as e:
        print(f"  [Live search error: {e}]")
        return []


def should_skip_rag(query: str) -> bool:
    """Detect queries that don't need RAG (greetings, chitchat, meta questions)."""
    chitchat = [
        "how are you", "who are you", "what can you do", "help",
        "what are your capabilities", "tell me about yourself"
    ]
    return any(phrase in query.lower() for phrase in chitchat)


def retrieve_context(query: str) -> tuple[str, str]:
    """Retrieve relevant passages using layered search."""
    if should_skip_rag(query):
        return "", "none"

    words = query.strip().split()
    if len(words) <= 2:
        return "", "none"

    local_results = search_local_index(query, TOP_K)
    best_local_score = max((s for _, s in local_results), default=0)

    live_results = []
    if best_local_score < 0.35 and live_search_available:
        live_results = search_huggingface_live(query, TOP_K)

    all_results = []
    for passage, score in local_results:
        all_results.append((passage, score, "local"))

    for passage, score in live_results:
        if not any(passage[:100] == existing[:100] for existing, _, _ in all_results):
            all_results.append((passage, score, "live"))

    all_results.sort(key=lambda x: x[1], reverse=True)
    top_results = all_results[:TOP_K]

    if not top_results:
        return "", "none"

    context_parts = []
    sources = set()
    for i, (passage, score, source) in enumerate(top_results):
        context_parts.append(f"[{i+1}] {passage}")
        sources.add(source)

    source_label = " + ".join(sorted(sources))
    return "\n\n".join(context_parts), source_label

wordsegment.load()
# Cache vocab as set for faster lookup
WORDSEG_VOCAB = set(wordsegment.WORDS)

def validate_response(response: str, query: str) -> tuple[str, bool]:
    """Check if response shows signs of hallucination/uncertainty."""
    # Red flags for hallucination
    hallucination_phrases = [
        "according to", "in 2026", "in 2025", "recently announced",
        "just released", "breaking news", "last week", "yesterday"
    ]
    
    # Check if discussing future/very recent events
    has_temporal_warning = any(phrase in response.lower() for phrase in hallucination_phrases)
    
    # Check if query asks about future
    future_keywords = ["2025", "2026", "latest", "recent", "current", "now", "today"]
    asks_about_future = any(kw in query.lower() for kw in future_keywords)
    
    if has_temporal_warning or asks_about_future:
        warning = "\n\n  Note: This response may contain speculative or outdated information. My training data has a cutoff date."
        return response + warning, True
    
    return response, False

def estimate_tokens_needed(query: str) -> int:
    """Adjust max_tokens based on query complexity."""
    words = len(query.split())
    
    # Short factual questions
    if words <= 10 and any(q in query.lower() for q in ["what is", "who is", "when", "where"]):
        return 256
    # Complex explanations
    elif any(word in query.lower() for word in ["explain", "describe", "how does", "mechanism"]):
        return 512
    # Default
    else:
        return 384

def generate_response(prompt: str) -> tuple[str, str]:
    """Generate a response, augmented with retrieved context if available."""
    context, source = retrieve_context(prompt)

    if context:
        # Simpler Llama-2 style prompt
        full_prompt = (
            f"Use the following context to answer the question. "
            f"If you cannot answer based on the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n"
            f"Answer:"
        )
    else:
        full_prompt = f"Question: {prompt}\nAnswer:"

    # Dynamic Token ALlocation based on query complexity
    max_tokens = estimate_tokens_needed(prompt)
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048 - max_tokens
    ).to(device)

    # Generate with OOM fallback (no streaming until spacing is fixed in training)
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
    except torch.cuda.OutOfMemoryError:
        print("  [VRAM full, retrying with shorter output...]")
        torch.cuda.empty_cache()
        inputs = tokenizer(full_prompt, return_tensors="pt", 
                          truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

    # Decode and fix spacing
    input_length = inputs.input_ids.shape[1]
    raw_response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    def segment_run(match):
        run = match.group(0)
        # Don't segment all-caps acronyms (any length if all caps)
        if run.isupper():
            return run
        if run.lower() in WORDSEG_VOCAB:
            return run
        return ' '.join(wordsegment.segment(run))

    # Apply word segmentation to the final result
    response = re.sub(r'[a-zA-Z]{9,}', segment_run, raw_response)
    
    # Cleaning  metadata artifacts from scraped web content
    cleanup_patterns = [
        r'\[Reference:.*?\]',           # Remove [Reference:[1]]
        r'\|answered by\|.*?\|',        # Remove |answered by|name|
        r'\|date created\|.*?\|',       # Remove |date created|date|
        r'\|last updated\|.*?\|',       # Remove |last updated|date|
        r'\|[Cc]omments\|.*',           # Remove |Comments|None|
        r'answered by:.*?(?=\n|$)',    # Remove "answered by: name"
        r'date created:.*?(?=\n|$)',   # Remove "date created: date"
        r'last updated:.*?(?=\n|$)',   # Remove "last updated: date"
        r'\bComments:.*?(?=\n|$)',     # Remove "Comments: None"
    ]
    
    for pattern in cleanup_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
    
    # Collapse multiple spaces/newlines and clean up
    response = re.sub(r'\s+', ' ', response).strip()
    response = re.sub(r'\s*\|\s*$', '', response)  # Remove trailing pipes
    
    return response.strip(), source


if __name__ == "__main__":
    # Chat Loop
    print("\n" + "=" * 60)
    print("  InfoSage-13B Chatbot")
    features = []
    if local_rag_available:
        features.append("Local RAG index")
    if live_search_available:
        features.append("Live HuggingFace search")
    if features:
        print(f"  Knowledge: {' + '.join(features)}")
    else:
        print("  Mode: Generation only (no RAG)")
    print("  Type 'exit' to quit")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print("Exiting chatbot...")
                break

            response, source = generate_response(prompt)
            response, warned = validate_response(response, prompt)

            if source != "none":
                print(f"  [Source: {source}]")
            if warned:
                print("  [Warning: Uncertain information - verify independently]")
            print(f"Bot: {response}\n")
        except KeyboardInterrupt:
            print("\nExiting chatbot...")
            break
        except Exception as e:
            print(f"\nError: {e}")
