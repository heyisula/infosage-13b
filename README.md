# FineWeb-Edu LLM Training

A complete pipeline for training a mini-LLM (GPT-2 architecture) from scratch using the High-Quality FineWeb-Edu dataset.

## 🚀 Project Overview

This repository contains a Jupyter-based training pipeline for a compact transformer model. The goal is to demonstrate the end-to-end process of data materialization, tokenizer training, model configuration, and distributed training on GPUs.

### Key Features:
- **Custom Tokenizer**: BPE-based tokenizer trained specifically on the FineWeb-Edu corpus.
- **Optimized Architecture**: A lightweight GPT-2 configuration designed for fast iteration and specific educational content.
- **Efficient Training**: Utilizes `fp16` mixed-precision and gradient accumulation to perform effective training on consumer-grade GPUs (e.g., RTX 4060).

## 🛠 Technical Specifications

### Model Architecture (GPT-2)
- **Vocabulary Size**: 50,000
- **Context Length**: 512 tokens
- **Embedding Dimension**: 512
- **Layers**: 8
- **Attention Heads**: 8

### Dataset: FineWeb-Edu
The model is trained on a materialized subset of **200,000 samples** from the [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset, which contains high-quality educational web content.

## 📈 Training Configuration
The latest training run completed **12,500 steps** (1 full epoch) with the following parameters:
- **Learning Rate**: 5e-5
- **Effective Batch Size**: 16 (2 per device × 8 gradient accumulation steps)
- **Precision**: FP16 mixed precision
- **Final Training Loss**: ~4.41

## 📂 Project Structure

- `train.ipynb`: The main notebook containing the full training pipeline.
- `chat_llm.py`: Script for interacting with the trained model (Utility/CLI).
- `out/`: Directory containing checkpoints and the trained tokenizer (ignored by git).
- `.gitignore`: Configured to exclude large model shards and temporary files.

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/heyisula/fineweb-edu-llm-training.git
   cd fineweb-edu-llm-training
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch datasets transformers tokenizers tqdm
   ```

3. **Train the Model**:
   Open `train.ipynb` in your Jupyter environment and run all cells. Ensure CUDA is available for GPU acceleration.

## 📝 License
This project is licensed under the MIT License.
