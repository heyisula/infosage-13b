"""
GUI Server for Llama-2-13B Chatbot

Launches a Flask web server with a premium chat interface.
Imports chat_llm.py as a module without modifying it.

Usage:
    python gui/app.py
"""

import sys
import os

# Adding project root to path to import chat_llm.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import chat_llm
import logging
logging.disable(logging.NOTSET) #Enabling logging for debugging


from flask import Flask, render_template, request, jsonify
import threading
import uuid
import json
import gc
from datetime import datetime
import glob
import multiprocessing
import logging

# Configure logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# Explicitly set paths to avoid ambiguity when running from adjacent directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            static_folder=os.path.join(BASE_DIR, 'static'),
            template_folder=os.path.join(BASE_DIR, 'templates'))

# Model state
model_state = {"status": "stopped", "error": None}
model_lock = threading.Lock()

# Chat history directory
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)


def patch_adapter_config(adapter_dir):
    """Remove unknown LoraConfig keys caused by PEFT version mismatch."""
    import inspect
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
    backup_path = config_path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    for k in unknown_keys:
        del config[k]

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("[patch] adapter_config.json patched.")


def load_model():
    """Load the model, tokenizer, embedder, and RAG index into chat_llm globals."""
    global model_state

    with model_lock:
        if model_state["status"] != "stopped":
            return
        model_state = {"status": "loading", "error": None}

    try:
        print("Starting model load...")
        import torch
        # logging.info(f"Torch imported. Cuda available: {torch.cuda.is_available()}")
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        # logging.info("Imports complete.")

        chat_llm.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {chat_llm.device}")

        # Tokenizer
        print(f"Loading base model tokenizer: {chat_llm.BASE_MODEL_ID}")
        chat_llm.tokenizer = AutoTokenizer.from_pretrained(
            chat_llm.BASE_MODEL_ID,
            clean_up_tokenization_spaces=True
        )
        chat_llm.tokenizer.pad_token = chat_llm.tokenizer.eos_token
        chat_llm.tokenizer.padding_side = "left"
        # logging.info("Tokenizer loaded.")

        # Patch adapter configuration to fix PEFT version mismatch issues (if any)
        patch_adapter_config(chat_llm.ADAPTER_DIR)

        # Quantization configuration for 4-bit loading using bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Base model
        max_memory = {0: "7500MiB"}
        print("Loading base model with quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            chat_llm.BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )
        # logging.info("Base model loaded.")

        # LoRA adapters
        print(f"Applying LoRA adapters from: {chat_llm.ADAPTER_DIR}")
        chat_llm.model = PeftModel.from_pretrained(base_model, chat_llm.ADAPTER_DIR)
        chat_llm.model.eval()
        # logging.info("LoRA adapters applied.")

        # Embedding model
        try:
            from sentence_transformers import SentenceTransformer
            chat_llm.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            chat_llm.embedder.num_proc = 1
            print("Embedding model loaded")
        except ImportError:
            print("sentence-transformers not installed. RAG disabled.")

        # FAISS index
        if chat_llm.embedder:
            try:
                import faiss
                index_path = os.path.join(chat_llm.RAG_DIR, "faiss_index.bin")
                if os.path.exists(index_path):
                    print("Loading local RAG knowledge base...")
                    chat_llm.faiss_index = faiss.read_index(index_path)
                    chat_llm.local_passages = np.load(
                        os.path.join(chat_llm.RAG_DIR, "passages.npy"),
                        allow_pickle=True
                    )
                    chat_llm.local_rag_available = True
                    print(f"Local RAG loaded: {chat_llm.faiss_index.ntotal:,} passages indexed")
            except ImportError:
                print("faiss-cpu not installed. Local RAG disabled.")

        # Live search capability from Fine Edu Hugging Face Dataset
        if chat_llm.LIVE_SEARCH_ENABLED and chat_llm.embedder:
            try:
                from datasets import load_dataset
                chat_llm.live_search_available = True
                print("Live HuggingFace search: ENABLED")
            except ImportError:
                print("datasets not installed. Live search disabled.")

        with model_lock:
            model_state = {"status": "ready", "error": None}
        # logging.info("Model ready!")
        print("Model ready!")

    except Exception as e:
        # logging.error(f"Model load failed: {e}", exc_info=True)
        print(f"Model load failed: {e}")
        with model_lock:
            model_state = {"status": "stopped", "error": str(e)}


def unload_model():
    """Unload the model and free VRAM."""
    global model_state
    import torch

    chat_llm.model = None
    chat_llm.tokenizer = None
    chat_llm.embedder = None
    chat_llm.faiss_index = None
    chat_llm.local_passages = None
    chat_llm.local_rag_available = False
    chat_llm.live_search_available = False

    # Forcing garbage collection to release Python objects
    gc.collect()
    
    # Empty CUDA cache to release VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    with model_lock:
        model_state = {"status": "stopped", "error": None}
    print("Model unloaded, VRAM freed.")


# Routes

@app.route("/")
def index():
    import random
    return render_template("index.html", version=random.randint(1, 10000))


@app.route("/api/model/status")
def get_model_status():
    # logging.info("Check status...") # Too spammy, maybe only errors?
    info = dict(model_state)
    # Add GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.mem_get_info()
            info["vram_free_mb"] = round(mem[0] / 1024 / 1024)
            info["vram_total_mb"] = round(mem[1] / 1024 / 1024)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["model_name"] = "Llama-2 13B (FineWeb)"
    except Exception as e:
        # logging.error(f"Status check error: {e}")
        pass
    return jsonify(info)


@app.route("/api/model/start", methods=["POST"])
def start_model():
    if model_state["status"] != "stopped":
        return jsonify({"error": "Model is not stopped"}), 400
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    return jsonify({"status": "loading"})


@app.route("/api/model/stop", methods=["POST"])
def stop_model():
    if model_state["status"] != "ready":
        return jsonify({"error": "Model is not running"}), 400
    unload_model()
    return jsonify({"status": "stopped"})


@app.route("/api/chat", methods=["POST"])
def chat():
    if model_state["status"] != "ready":
        return jsonify({"error": "Model is not loaded. Start the model first."}), 503

    data = request.json
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        response, source = chat_llm.generate_response(message)
        return jsonify({"response": response, "source": source})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def list_history():
    chats = []
    for filepath in sorted(glob.glob(os.path.join(HISTORY_DIR, "*.json")), reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            chats.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "created": data.get("created", ""),
                "messageCount": len(data.get("messages", []))
            })
        except:
            continue
    return jsonify(chats)


@app.route("/api/history/<chat_id>")
def get_history(chat_id):
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if not os.path.exists(filepath):
        return jsonify({"error": "Not found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/history", methods=["POST"])
def save_history():
    data = request.json
    chat_id = data.get("id") or str(uuid.uuid4())
    data["id"] = chat_id
    if "created" not in data:
        data["created"] = datetime.now().isoformat()

    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return jsonify({"id": chat_id})


@app.route("/api/history/<chat_id>", methods=["DELETE"])
def delete_history(chat_id):
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    return jsonify({"ok": True})


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("\n" + "=" * 50)
    print("  InfoSage AI - 13B")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    logging.info("Server starting...")
    app.run(host="0.0.0.0", port=5000, debug=False)
