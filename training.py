# %%
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from shared import xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, get_gsm8k_questions, correctness_reward_func

# %%
dataset = get_gsm8k_questions("train")

# %%
model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
# Load LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# %%
max_prompt_length = 256
max_seq_length = 1024

training_args = GRPOConfig(
    learning_rate=5e-6,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=8, 
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,    
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=1, 
    bf16=True,
    report_to="tensorboard",
    output_dir="outputs",
    save_steps=100,
    logging_steps=2,
    max_grad_norm=0.1,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# %%
trainer.train()

# %%
