<div align="center">

# 🧠 FineWeb-Edu LLM Training

**Production-grade QLoRA fine-tuning of Llama-2-13B on educational web content — with a RAG-powered chatbot built in.**

[![Model](https://img.shields.io/badge/Model-Llama--2--13B-blueviolet)](https://huggingface.co/NousResearch/Llama-2-13b-hf)
[![Dataset](https://img.shields.io/badge/Dataset-FineWeb--Edu-blue)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
[![GPU](https://img.shields.io/badge/GPU-H100%2080GB-green)](https://www.nvidia.com/en-us/data-center/h100/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 👋 What is this?

This project takes **Meta's Llama-2 13B** model and fine-tunes it on **1 million high-quality educational passages** from the FineWeb-Edu dataset. The result is a language model that's better at explaining concepts, answering questions, and holding educational conversations.

On top of that, there's a **RAG chatbot** (`chat_llm.py`) that doesn't just rely on what the model "remembers" — it actively searches a local knowledge base to back up its answers with real passages.

Think of it as a smarter tutor that can both reason *and* look things up.

---

## ✨ Key Features

| Feature | Details |
| :--- | :--- |
| 🦙 **Llama-2-13B** | 13 billion parameter base model from Meta |
| ⚡ **QLoRA** | 4-bit NF4 quantization — trains a 13B model on a single GPU |
| 🚀 **H100 Optimized** | Flash Attention 2, BF16, TF32, batch 16, no gradient checkpointing |
| 📊 **Live Diagnostics** | Real-time it/s, tok/s, and ETA monitoring during training |
| 📚 **1M Samples** | Streamed from FineWeb-Edu (never loads full dataset into RAM) |
| 🔍 **RAG Chat** | FAISS vector search + live HuggingFace fallback |
| 💾 **Auto-Resume** | Checkpoints save to Google Drive; training resumes if Colab disconnects |
| 🛡️ **OOM Safety** | Graceful error handling with diagnostic output on memory failures |

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
                          RAG Chatbot
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
  <tr><td><b>Sequence Length</b></td><td>2,048 tokens</td></tr>
  <tr><td><b>Batch Size</b></td><td>16 (no gradient accumulation)</td></tr>
  <tr><td><b>Optimizer</b></td><td>Paged AdamW 32-bit</td></tr>
  <tr><td><b>LR Schedule</b></td><td>Cosine (1e-4, 3% warmup)</td></tr>
  <tr><td><b>Precision</b></td><td>BFloat16 + TF32</td></tr>
  <tr><td><b>Attention</b></td><td>Flash Attention 2</td></tr>
  <tr><td><b>Grad Checkpointing</b></td><td>Disabled (H100 has headroom)</td></tr>
  <tr><td><b>Dataloader</b></td><td>8 workers, persistent, pinned, drop_last</td></tr>
  <tr><td><b>Max Steps</b></td><td>5,000 (~45–60 min on H100)</td></tr>
  <tr><td><b>Hardware</b></td><td>NVIDIA H100 80GB HBM3</td></tr>
</table>

**Expected throughput**: ~1.2–1.6 it/s on H100.

---

## 🔐 Authentication

This project uses the **NousResearch/Llama-2-13b-hf** community mirror, which is **fully open and ungated** — no HuggingFace token or license acceptance is required. Just run the notebook and it downloads automatically.

> 💡 **Want to use the official Meta model instead?** Change `MODEL_NAME` in the notebook to `meta-llama/Llama-2-13b-hf`. You'll need to:
> 1. Accept the license at [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
> 2. Create a [HuggingFace token](https://huggingface.co/settings/tokens) with Read access
> 3. Add it as `HF_TOKEN` in your Colab secrets (🔑 icon in the sidebar)

---

## 📂 Project Structure

```
fineweb-edu-llm-training/
├── train.ipynb          # Fine-tuning notebook (Colab-ready, H100-optimized)
├── chat_llm.py          # RAG chatbot with layered retrieval
├── build_rag_index.py   # Standalone FAISS index builder
├── README.md            # You are here
└── out/                 # Model checkpoints & RAG index (git-ignored)
```

---

## 🚀 Getting Started

### Cloud Training (Recommended)

1. Upload `train.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set the runtime to **H100 GPU** (or A100 if H100 isn't available)
3. Run all cells — hardware diagnostics will confirm your setup
4. Model and RAG index are saved to your Google Drive automatically

### Local Chat

Once you've trained the model:

```bash
# 1. Download from Google Drive
#    → fineweb_edu_llama2_13b/final_model/
#    → fineweb_edu_llama2_13b/rag_index/

# 2. Install dependencies
pip install torch transformers datasets faiss-cpu sentence-transformers peft bitsandbytes accelerate

# 3. Start chatting
python chat_llm.py
```

---

## 🤝 Contributing

Found a bug? Have an idea? Feel free to open an issue or submit a PR. All contributions are welcome.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ using HuggingFace Transformers, PEFT, and a lot of GPU hours.</sub>
</div>
