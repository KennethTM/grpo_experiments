# %%
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from shared import get_gsm8k_questions, messages_to_text, save_predictions_to_parquet
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"

# %%
model = LLM(model_id, 
            dtype="bfloat16", 
            max_num_seqs = 64, 
            max_seq_len_to_capture=1024, 
            enable_lora=True)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    max_tokens=1024,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
# Test data
dataset = get_gsm8k_questions("test")
dataset = dataset.map(messages_to_text, fn_kwargs={"tokenizer": tokenizer})

# %%
# Predictions with base model
predictions = model.generate(dataset["text"], sampling_params=sampling_params)

# %%
results_df = save_predictions_to_parquet(dataset, 
                                         predictions, 
                                         "results/qwen3_06b_gsm8k.parquet")

# %%
print(f"Base model accuracy: {(results_df['predicted_answer'] == results_df['true_answer']).mean():.3f}")
print(f"Base model stop finish: {(results_df['finish_reason'] == 'stop').mean():.3f}")

# %%
lora_path = "outputs/checkpoint-1100"
predictions_lora = model.generate(dataset["text"], 
                                  sampling_params=sampling_params,
                                  lora_request=LoRARequest("lora", 1, lora_path))

# %%
lora_results_df = save_predictions_to_parquet(dataset, 
                                              predictions_lora, 
                                              "results/qwen3_06b_lora_gsm8k.parquet")

# %%
print(f"LoRA model accuracy: {(lora_results_df['predicted_answer'] == lora_results_df['true_answer']).mean():.3f}")
print(f"LoRA model stop finish: {(lora_results_df['finish_reason'] == 'stop').mean():.3f}")

# %%