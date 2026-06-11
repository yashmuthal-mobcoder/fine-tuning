# Medical LLM Fine-Tuning Pipeline

A complete end-to-end project for fine-tuning Large Language Models (LLMs) on medical Question-Answer datasets using LoRA and MLX-based QLoRA-style training.

The project supports:

* Dataset preprocessing
* Chat-template tokenization
* Hugging Face LoRA fine-tuning
* MLX quantized training on Apple Silicon
* Evaluation and inference
* Dynamic model selection (Qwen, Phi, Llama)

---

# Features

* Fine-tune instruction models on medical datasets
* LoRA-based PEFT training
* Quantized MLX training for Apple Silicon
* Dynamic model selection
* Train/eval dataset generation
* Interactive inference
* Modular project structure

---

# Project Structure

```text
fine_tuning/
│
├── data/
│   ├── medquad_cleaned.csv
│   ├── train_dataset/
│   └── eval_dataset/
│
├── dataset/
│
├── training/
│   ├── config.py
│   ├── config.yaml
│   ├── tokenizer.py
│   └── fine_tune.py
│
├── testing/
│   ├── chat.py
│   ├── test.py
│   ├── test1.py
│   └── test2.py
│
├── mlx_data/
│
├── medical_adapter/
├── medical_adapter_v2/
├── medical_qwen_lora/
│
├── README.md
└── .gitignore
```

---

# Fine-Tuning Overview

## Full Fine-Tuning

Updates all model parameters.

### Advantages

* Maximum adaptability
* Highest potential performance

### Disadvantages

* Expensive
* Large memory requirements
* Large checkpoints

---

## LoRA

Low-Rank Adaptation freezes base model weights and trains small adapter matrices.

### Advantages

* Fast training
* Small checkpoints
* Low memory consumption

### Disadvantages

* Slightly lower performance than full fine-tuning

---

## QLoRA

Quantized LoRA training.

Workflow:

1. Quantize model to 4-bit
2. Freeze quantized weights
3. Train LoRA adapters

### Advantages

* Extremely memory efficient
* Faster training
* Enables larger models on consumer hardware

---

# Dataset

Dataset used:

**MedQuAD**

Contains:

* Medical Questions
* Medical Answers

Example:

```json
{
  "question": "What is glaucoma?",
  "answer": "Glaucoma is a group of eye diseases that damage the optic nerve."
}
```

---

# Data Preprocessing

The preprocessing pipeline:

1. Load CSV dataset
2. Create chat-style conversations

```python
[
    {"role": "system"},
    {"role": "user"},
    {"role": "assistant"}
]
```

3. Apply model chat template
4. Generate:

* input_ids
* attention_mask
* labels

5. Mask prompt tokens with:

```python
-100
```

6. Split into train and evaluation datasets

---

# Environment Setup

## Hugging Face Environment

```bash
python -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch transformers datasets peft accelerate pandas pyyaml sentencepiece
```

---

## MLX Environment

```bash
python -m venv mlx-env

source mlx-env/bin/activate
```

Install MLX:

```bash
pip install mlx mlx-lm
```

Verify:

```bash
mlx_lm.generate --help
```

---

# Supported Models

Configure models inside:

```text
training/config.yaml
```

Example:

```yaml
models:
  qwen:
    hf_name: "Qwen/Qwen2.5-1.5B-Instruct"

  phi:
    hf_name: "microsoft/Phi-3-mini-4k-instruct"

  llama:
    hf_name: "meta-llama/Llama-3.2-1B-Instruct"
```

---

# Dataset Tokenization

Generate train and evaluation datasets:

```bash
python training/tokenizer.py --model qwen
```

or

```bash
python training/tokenizer.py --model phi
```

or

```bash
python training/tokenizer.py --model llama
```

Outputs:

```text
data/train_dataset
data/eval_dataset
```

---

# Hugging Face LoRA Training

Train LoRA adapters:

```bash
python training/fine_tune.py --model qwen
```

```bash
python training/fine_tune.py --model phi
```

```bash
python training/fine_tune.py --model llama
```

Output:

```text
medical_qwen_lora/
```

Contains:

```text
adapter_model.safetensors
adapter_config.json
```

---

# Testing Hugging Face Model

Launch interactive chat:

```bash
python testing/chat.py --model qwen
```

Example:

```text
You: What is glaucoma?

Assistant:
Glaucoma is a group of eye diseases that damage the optic nerve...
```

---

# MLX Model Conversion

Convert Hugging Face model to quantized MLX format:

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-1.5B-Instruct \
  --mlx-path qwen-medical-4bit \
  -q
```

Output:

```text
qwen-medical-4bit/
```

---

# MLX Dataset Format

Directory structure:

```text
mlx_data/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

Example:

```json
{
  "prompt": "What is glaucoma?",
  "completion": "Glaucoma is a group of eye diseases..."
}
```

---

# MLX LoRA Training

Train quantized model:

```bash
mlx_lm.lora \
  --model qwen-medical-4bit \
  --train \
  --data mlx_data \
  --fine-tune-type lora \
  --batch-size 1 \
  --iters 100 \
  --learning-rate 1e-4 \
  --adapter-path medical_adapter
```

Output:

```text
medical_adapter/
```

Contains:

```text
adapters.safetensors
adapter_config.json
```

---

# MLX Inference

## Base Model

```bash
mlx_lm.generate \
  --model qwen-medical-4bit \
  --prompt "What is glaucoma?"
```

## Fine-Tuned Model

```bash
mlx_lm.generate \
  --model qwen-medical-4bit \
  --adapter-path medical_adapter \
  --prompt "What is glaucoma?"
```

---

# Evaluation

The project supports comparison between:

* Base model
* LoRA model
* Quantized LoRA model

Metrics:

* Evaluation Loss
* Perplexity
* Response Quality

---

# Common Issues

## NaN Loss

Possible causes:

* High learning rate
* Incorrect labels
* Invalid token IDs

Recommended:

```python
learning_rate = 1e-4
max_grad_norm = 1.0
```

---

## MLX Out Of Memory

Reduce:

```bash
--batch-size
```

or

```bash
--num-layers
```

or

```bash
--max-seq-length
```

---

## Hugging Face Authentication Error

Login:

```bash
huggingface-cli login
```

---

# Future Improvements

* Retrieval Augmented Generation (RAG)
* Medical Knowledge Graph Integration
* Structured JSON Responses
* Automated Benchmarking
* Multi-Domain Fine-Tuning
* Full QLoRA Benchmarking
* Model Registry Support
* FastAPI Inference API

---

# License

MIT License
