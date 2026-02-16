# FineWeb-Edu LLM Training

A complete pipeline for fine-tuning **GPT-2 Medium (355M params)** on the FineWeb-Edu dataset, with a **RAG-enhanced chatbot** that retrieves relevant educational content in real-time.

## 🚀 Project Overview

This repository contains a Jupyter-based fine-tuning pipeline and a chatbot with Retrieval-Augmented Generation (RAG). The model is fine-tuned on high-quality educational web content and can search through its knowledge base to provide grounded answers.

### Key Features:
- **Fine-Tuned GPT-2 Medium**: 355M parameter pre-trained model, fine-tuned on educational content.
- **VRAM Optimized**: Gradient checkpointing + Adafactor + FP16 — trains on consumer GPUs (8GB VRAM).
- **RAG-Enhanced Chat**: FAISS vector search retrieves relevant passages from FineWeb-Edu in real-time.

## 🛠 Technical Specifications

### Model Architecture (GPT-2 Medium)
- **Parameters**: ~355 Million
- **Vocabulary Size**: 50,257 (GPT-2 pre-trained tokenizer)
- **Context Length**: 1,024 tokens
- **Embedding Dimension**: 1,024
- **Layers**: 24
- **Attention Heads**: 16

### Dataset: FineWeb-Edu
The model is fine-tuned on **500,000 samples** from [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a curated dataset of high-quality educational web content.

### RAG Pipeline (Layered Retrieval)
- **Layer 1 — Local FAISS Index**: Pre-built vector store with ~100K passages (instant, offline)
- **Layer 2 — Live HuggingFace Search**: Streams FineWeb-Edu in real-time, filters by keywords, ranks by embedding similarity (requires internet)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval**: Top-3 most relevant passages per query

## 📈 Training Configuration
- **Base Model**: `gpt2-medium` (pre-trained)
- **Learning Rate**: 2e-5 (with 500-step warmup)
- **Effective Batch Size**: 16 (1 per device × 16 gradient accumulation)
- **Precision**: FP16 mixed precision
- **Optimizer**: Adafactor (memory-efficient)
- **Gradient Checkpointing**: Enabled (~40% VRAM savings)

## 📂 Project Structure

- `train.ipynb`: Fine-tuning pipeline (data loading, tokenization, training, RAG index building).
- `chat_llm.py`: RAG-enhanced chatbot for interacting with the fine-tuned model.
- `build_rag_index.py`: Standalone script to build/rebuild the FAISS knowledge base.
- `out/`: Directory containing model checkpoints, tokenizer, and RAG index (git-ignored).
- `.gitignore`: Configured to exclude large model shards and temporary files.

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/heyisula/fineweb-edu-llm-training.git
   cd fineweb-edu-llm-training
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch datasets transformers tqdm
   pip install faiss-cpu sentence-transformers  # for RAG
   ```

3. **Fine-Tune the Model**:
   Open `train.ipynb` in Jupyter and run all cells. Requires CUDA GPU with ≥8GB VRAM.

4. **Build RAG Index** (if not built during training):
   ```bash
   python build_rag_index.py
   ```

5. **Chat with the Model**:
   ```bash
   python chat_llm.py
   ```

## 📝 License
This project is licensed under the MIT License.
