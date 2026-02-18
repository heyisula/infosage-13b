<div align="center">

# 🧠 InfoSage

**Private, local AI interface powered by Llama-2-13B and FineWeb-Edu.**

[![Model](https://img.shields.io/badge/Model-Llama--2--13B-blueviolet)](https://huggingface.co/NousResearch/Llama-2-13b-hf)
[![Dataset](https://img.shields.io/badge/Dataset-FineWeb--Edu-blue)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
[![Train GPU](https://img.shields.io/badge/Training-H100%2080GB-green)](https://www.nvidia.com/en-us/data-center/h100/)
[![Inference GPU](https://img.shields.io/badge/Inference-RTX%204060%208GB-orange)](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-family/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 👋 What is this?

This project fine-tunes **Meta's Llama-2 13B** on **1 million educational passages** from the FineWeb-Edu dataset. The goal is a model that's better at explaining things and answering questions.

Training happens on an H100 in the cloud, but the resulting model runs locally on consumer GPUs with as little as 8GB VRAM using 4-bit quantization.

There's also the **InfoSage RAG chatbot** (`chat_llm.py`) that doesn't just rely on what the model memorized during training. It searches a local FAISS knowledge base first, and if results aren't good enough, it falls back to a live search on HuggingFace.

---

## ✨ Key Features

| Feature | Details |
| :--- | :--- |
| 🦙 **Llama-2-13B** | 13 billion parameter base model from Meta |
| ⚡ **QLoRA** | 4-bit NF4 quantization, trains a 13B model on a single GPU |
| 🚀 **H100 Optimized** | Flash Attention 2, BF16, TF32, Batch 2 / Accum 4 |
| 🖥️ **Runs on 8GB GPUs** | Local inference with SDPA attention and precise VRAM mapping |
| 🛡️ **Grad Checkpointing** | Reduces activation VRAM from 40GB to around 4GB |
| 📊 **Live Diagnostics** | Real-time it/s, tok/s, and ETA monitoring during training |
| 📚 **1M Samples** | Streamed from FineWeb-Edu (never loads full dataset into RAM) |
| 🔍 **RAG Chat** | FAISS vector search + live HuggingFace fallback |
| 💾 **Auto-Resume** | Checkpoints save to Google Drive; training resumes if Colab disconnects |
| 🧹 **Memory Cleanup** | Throttled MemoryCallback (every 50 steps) for max throughput |
| 🎨 **Liquid Glass UI** | Premium web interface with dark/light mode and VRAM monitoring |
| ⚡ **Smart Controls** | One-click model loading/unloading with real-time status feedback |

---

## 🏗️ How It Works

```
FineWeb-Edu (1M samples)
        │
        ▼
   Streaming Tokenizer ──► QLoRA Trainer (H100)
                                  │
                                  ▼
                          LoRA Adapters (saved to Drive)
                                  │
                                  ▼
                          RAG Chatbot (Llama-2-13B)
                         ┌─────────────┐
                         │ FAISS Index  │ ◄── 100K passages
                         │ (local)      │
                         ├─────────────┤
                         │ HuggingFace │ ◄── live cloud search
                         │ (fallback)   │
                         └─────────────┘
```

---

## 📊 Training Configuration

<table>
  <tr><td><b>Base Model</b></td><td><code>NousResearch/Llama-2-13b-hf</code></td></tr>
  <tr><td><b>Quantization</b></td><td>4-bit NF4 + Double Quantization</td></tr>
  <tr><td><b>LoRA Rank</b></td><td>r=32, alpha=64, bias=none</td></tr>
  <tr><td><b>LoRA Targets</b></td><td><code>q_proj</code>, <code>k_proj</code>, <code>v_proj</code>, <code>o_proj</code></td></tr>
  <tr><td><b>Sequence Length</b></td><td>1,024 tokens (Optimized for VRAM)</td></tr>
  <tr><td><b>Batch Size</b></td><td>2 per device (Gradient Accumulation 4)</td></tr>
  <tr><td><b>Optimizer</b></td><td>AdamW 8-bit (bitsandbytes)</td></tr>
  <tr><td><b>LR Schedule</b></td><td>Cosine (1e-4, 150 warmup steps)</td></tr>
  <tr><td><b>Precision</b></td><td>BFloat16 + TF32</td></tr>
  <tr><td><b>Attention</b></td><td>Flash Attention 2 (training) / SDPA (local inference)</td></tr>
  <tr><td><b>Grad Checkpointing</b></td><td>Enabled (Required for 13B on 80GB)</td></tr>
  <tr><td><b>Dataloader</b></td><td>4 workers, persistent, pinned, drop_last</td></tr>
  <tr><td><b>Max Steps</b></td><td>5,000</td></tr>
  <tr><td><b>Hardware</b></td><td>NVIDIA H100 80GB HBM3</td></tr>
</table>

**Expected throughput**: ~1.1-1.3 it/s on H100 with checkpointing enabled.

---

## 🖥️ Local Inference

The chatbot is designed to run on consumer GPUs. Here's what makes that possible:

- **4-bit quantization** keeps the 13B model under 8GB VRAM
- **SDPA attention** is used instead of Flash Attention 2 (which is hard to install on Windows)
- **Precise memory mapping** (`7500MiB` GPU limit) leaves headroom for generation activations
- **Word segmentation** (`wordsegment` library) post-processes the output to fix spacing artifacts from fine-tuning
- **Windows-safe** multiprocessing guards prevent duplicate loading from spawned workers

Tested on an RTX 4060 (8GB) with 32GB system RAM.

---

## 🔐 Authentication

This project uses the **NousResearch/Llama-2-13b-hf** community mirror, which is fully open and ungated. No HuggingFace token or license acceptance needed. Just run the notebook and it downloads automatically.

---

## 📂 Project Structure

```
fineweb-edu-llm-training/
├── train.ipynb              # Fine-tuning notebook (H100-optimized)
├── gui/                     # InfoSage Web Interface (Flask + HTML/CSS/JS)
│   ├── app.py
│   ├── static/
│   └── templates/
├── chat_llm.py              # Llama-2-13B RAG chatbot
├── build_rag_index.py       # Standalone FAISS index builder
├── README.md                # Documentation
├── LICENSE                  # MIT License
├── .gitignore               # Git ignore rules
└── out/
    ├── final_model/         # LoRA adapters
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── training_args.bin
    └── rag_index/           # FAISS vector index
        ├── faiss_index.bin
        └── passages.npy
```

---

## 🚀 Getting Started

### Cloud Training (Recommended)

1. Upload `train.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set the runtime to **H100 GPU**
3. Run all cells. Hardware diagnostics will confirm your setup
4. Model and RAG index are saved to your Google Drive automatically

### Local Chat

Once you've trained the model and downloaded the files into the `out/` folder:

```bash
# 1. Install dependencies
pip install torch transformers datasets faiss-cpu sentence-transformers peft bitsandbytes accelerate wordsegment tqdm

# 2. Start the InfoSage Interface (Recommended)
python gui/app.py

# Or run the CLI chatbot
python chat_llm.py
```

The chatbot checks the local FAISS index first, and if results aren't good enough it falls back to a live search on HuggingFace.

### Rebuilding the RAG Index

If you want to rebuild or update the local knowledge base separately:

```bash
python build_rag_index.py
```

This streams 100K samples from FineWeb-Edu, chunks them into passages, embeds them, and writes the FAISS index to `out/rag_index/`.

---

## 🤝 Contributing

Found a bug? Have an idea? Feel free to open an issue or submit a PR. All contributions are welcome.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ using HuggingFace Transformers, PEFT, and bitsandbytes.</sub>
</div>
