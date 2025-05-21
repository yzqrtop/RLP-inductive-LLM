# Inductive thinking in generation model

## 📢 News
[2025.1] 🎉 We release our code for evaluation on three benchmarks.

## Introduction

We introduce **RLP**, a novel and versatile thought-augmented reasoning approach designed to enhance the accuracy, efficiency, and robustness of large language models (LLMs).

## Evaluation and Inference with Inductive thinking

### 1. Download Datasets 

Download inductive_thinking_LLM_dataset from ![Arrebol-yzq]{https://huggingface.co/datasets/Arrebol-yzq/inductive_thinking_LLM_dataset}. And put it in folder data.

### 2. Quick Start

First, set up the environment: https://github.com/yzqrtop/RLP-inductive-LLM.git

```bash
git clone 
cd RLP_inductive_LLM
conda create -n RLPLLM python==3.10 
conda activate RLPLLM
pip install -r requirements.txt
```

second, you can download model from ![RLP_LLM_Model]{https://huggingface.co/Arrebol-yzq/RLP_llm_inductive_model}. And put it in folder rl_model_by_llama_3_2_1B.
finally, you can run this RLP_LLM_Model by running the following scirpts.

```python
python train_ppo/validation_model.py # you can obtain a reward model. Code logic can be viewed from the contents of Python files
```
### 3. train PPO

Here we provide our inference code of  **IMAP** on BBH problems.  We provide some thought templates about problems in data(./preference_temp.json | ./queries_temp.json)
and, you can run this code to achieve the process of PPO.

```python
python train_ppo/validation_model.py # you can obtain a reward model. Code logic can be viewed from the contents of Python files
```

## 📖 BibTeX

