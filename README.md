# FineWeb-Edu LLM Training

A complete pipeline for fine-tuning **GPT-2 Large (774M params)** on the FineWeb-Edu dataset, with a **RAG-enhanced chatbot** that retrieves relevant educational content in real-time.

## 🚀 Project Overview

This repository contains a Jupyter-based fine-tuning pipeline and a chatbot with Retrieval-Augmented Generation (RAG). The model is fine-tuned on high-quality educational web content and can search through its knowledge base to provide grounded answers.

### Key Features:
- **QLoRA Fine-Tuning**: Efficiently fine-tune GPT-2 Large using 4-bit quantization + LoRA adapters.
- **Extreme VRAM Efficiency**: Fits a 774M model in ~5GB total VRAM (T4 optimized).
- **Layered RAG Chat**: Combines local FAISS vector search with live HuggingFace streaming search.

## 🛠 Technical Specifications

### Model Architecture (GPT-2 Large)
- **Parameters**: ~774 Million
- **Vocabulary Size**: 50,257
- **Layers**: 36
- **Attention Heads**: 20
- **Embedding Dimension**: 1280
- **Context Length**: 1,024 tokens

### Dataset: FineWeb-Edu
The model is fine-tuned on **1,000,000 samples** from [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

### RAG Pipeline (Layered Retrieval)
- **Layer 1 — Local FAISS Index**: Pre-built vector store with ~100K passages (instant, offline).
- **Layer 2 — Live HuggingFace Search**: Streams FineWeb-Edu in real-time from the cloud if local results are weak.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval**: Top-3 most relevant passages per query.

## 📈 Training Configuration
- **Base Model**: `gpt2-large`
- **Learning Rate**: 2e-5 (with 500-step warmup)
- **Effective Batch Size**: 8 (1 per device × 8 gradient accumulation)
- **Precision**: FP16 mixed precision
- **Optimizer**: Adafactor
- **Gradient Checkpointing**: Enabled (Essential for 15GB VRAM)

## 📂 Project Structure

- `train.ipynb`: Fine-tuning pipeline with Colab setup and Google Drive persistence.
- `chat_llm.py`: RAG-enhanced chatbot with layered retrieval.
- `build_rag_index.py`: Standalone script to build/rebuild the FAISS knowledge base.
- `out/`: Directory containing model checkpoints and RAG index (git-ignored).

## ⚙️ Installation & Usage

### 1. Cloud Training (Recommended)
1. Upload `train.ipynb` to **Google Colab**.
2. Set Runtime to **T4 GPU**.
3. Run the setup cells to mount Google Drive and install dependencies.
4. Run all cells to fine-tune and build the RAG index. Matches are saved to your Drive.

### 2. Local Chat
Once trained:
1. Download the `fineweb_edu_gpt2_large` folder from Drive to your local `out/` directory.
2. Install local deps:
   ```bash
   pip install torch transformers datasets faiss-cpu sentence-transformers
   ```
3. Run the chat:
   ```bash
   python chat_llm.py
   ```

## 📝 License
This project is licensed under the MIT License.
