<div align="center">

# 🧠 InfoSage AI
### Advanced Local Intelligence Engine for Educational Synthesis

**A high-performance interface for Llama-2-13B, fine-tuned on the FineWeb-Edu corpus.**

[![Model](https://img.shields.io/badge/Model-Llama--2--13B-3b82f6)](https://huggingface.co/NousResearch/Llama-2-13b-hf)
[![Dataset](https://img.shields.io/badge/Dataset-FineWeb--Edu-06b6d4)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
[![Training](https://img.shields.io/badge/Training-H100_80GB-22c55e)](https://www.nvidia.com/en-us/data-center/h100/)
[![Inference](https://img.shields.io/badge/Inference-RTX_4060_8GB-f97316)](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-family/)
[![UI](https://img.shields.io/badge/UX-Electric_Azure-0ea5e9)](https://github.com/heyisula/fineweb-edu-llm-training)
[![License](https://img.shields.io/badge/License-MIT-gray)](LICENSE)

---

**InfoSage** is a local-first AI ecosystem designed for high-fidelity educational content retrieval and synthesis. By combining Meta's **Llama-2-13B** architecture with a massive **1M sample fine-tune** from the FineWeb-Edu dataset, the system provides superior clarity and reasoning while ensuring total data privacy on consumer hardware.

</div>

---

## Technical Overview

### The Problem
Fine-tuning a 13-billion parameter model usually requires enterprise-grade clusters. Running such a model locally often results in slow inference or memory crashes on standard 8GB GPUs.

### The InfoSage Solution
We utilize a **Cloud-to-Local pipeline**:
1.  **Cloud Training**: High-speed optimization on an NVIDIA H100 using QLoRA.
2.  **Local Synthesis**: Deployment on consumer NVIDIA RTX cards using 4-bit NF4 quantization and SDPA attention mapping.
3.  **Hybrid RAG**: A dual-stage retrieval system that combines a local FAISS knowledge base with a live internet fallback.

---

## End-to-End Workflow

### Phase 1: Cloud Fine-Tuning
1.  Open `train.ipynb` in **Google Colab**.
2.  Set the runtime to **H100 GPU** (Required for 13B + Flash Attention 2).
3.  Mount Google Drive to persist checkpoints.
4.  Execute all cells. The script will stream 1M samples from FineWeb-Edu and output LoRA adapters.

### Phase 2: Local Deployment
1.  Download the `final_model` and `rag_index` folders from your Google Drive.
2.  Place them into the project `out/` directory:
    *   `out/final_model/` (contains adapters and tokenizer)
    *   `out/rag_index/` (contains FAISS index and passages)
3.  Install local dependencies:
    ```bash
    pip install torch transformers datasets faiss-cpu sentence-transformers peft bitsandbytes accelerate wordsegment tqdm
    ```

### Phase 3: Hardware Verification
The system is optimized for **Windows + NVIDIA RTX**.
*   **VRAM Management**: High-precision mapping (`7500MiB` limit) ensures 13B fits into an 8GB VRAM envelope.
*   **Post-Processing**: Uses the `wordsegment` library to fix spacing artifacts common in fine-tuned Llama models.

---

## Key Capabilities

### Advanced Architecture
*   **Hybrid RAG Pipeline**: Intelligent switching between a local vector store and live HuggingFace search results.
*   **4-Bit NormalFloat (NF4)**: Drastic memory reduction without significant loss in perplexity.
*   **Double Quantization**: Further reduces VRAM overhead for the quantization constants themselves.

### Electric Azure Design System
*   **Liquid Glass UI**: Premium frosted-glass panels with dynamic `backdrop-filter` depth.
*   **Hardware Telemetry**: Real-time VRAM monitoring and automatic GPU model detection.
*   **Spotlight Interactions**: Cursor-responsive lighting effects on all dashboard surfaces.

---

## Running the System

### Option 1: Premium Dashboard (Recommended)
The full web interface with ambient background and real-time diagnostics.
```bash
python gui/app.py
# Access: http://localhost:5000
```

### Option 2: Terminal Session
For developers who prefer a minimalist, low-latency CLI experience.
```bash
python chat_llm.py
```

### Option 3: Index Maintenance
Rebuild or update the local FAISS knowledge base with new data.
```bash
python build_rag_index.py
```

---

## Technical Specifications

| Component | Configuration |
| :--- | :--- |
| **Foundation Model** | Meta Llama-2-13B (NousResearch Mirror) |
| **Fine-Tuning Dataset** | FineWeb-Edu (1,000,000 samples) |
| **Training Compute** | NVIDIA H100 80GB HBM3 |
| **Inference Compute** | NVIDIA RTX Series (Minimum 8GB Dedicated VRAM) |
| **Quantization Type** | 4-Bit NF4 + Double Quantization |
| **LoRA Specs** | Rank 32 / Alpha 64 / Target: All Linear Layers |
| **Attention Policy** | Flash Attention 2 (Train) / SDPA (Local) |

---

## Project Governance
*   **`gui/`**: Full-stack application assets (Flask, Outfit font, Azure theme).
*   **`chat_llm.py`**: The "brain"—manages LLM loading, RAG routing, and post-processing.
*   **`build_rag_index.py`**: Data engineering tool for vector index generation.
*   **`train.ipynb`**: Notebook for high-performance cloud training.

---

<div align="center">
  <sub>InfoSage is an open-source research project licensed under the MIT framework.</sub>
</div>
