# GRPO experiments

## About

Experiments with RL post-training using small model (Qwen/Qwen3-0.6B) and simple math dataset (openai/gsm8k).

Model is already SFT tuned and with thinking traces, we only apply GRPO training using LORA adapters on top to improve performance (`training.py`).

Performance of "base model" and "lora model" is evaluated on test set (`inference.py`).

Training on just part of the dataset boost accuracy and improves the models ability to thinking using a constrained token budget.

## Example results

Example of performance gains:

Base model accuracy = 0.388 and stop rate = 0.824

LoRA model accuracy = 0.679 and stop rate = 0.935

The accuracy improves. Stop rate is how often the model successfully stops generation due to a end of sentence token, which also improves as the model is being trained to figure of the question using less tokens.

## Install

Install dependencies using uv (`uv sync`). 

Current setup/config runs with 16 GB VRAM GPU.

## Based on

Huggingface tutorials:

https://huggingface.co/learn/llm-course/en/chapter12/5?fw=pt

https://huggingface.co/learn/llm-course/en/chapter12/6?fw=pt

TRL docs:

https://huggingface.co/docs/trl/main/en/grpo_trainer
https://huggingface.co/docs/trl/main/en/vllm_integration

vLLM docs:

https://docs.vllm.ai/en/latest/usage/index.html

