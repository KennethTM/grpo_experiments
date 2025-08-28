from datasets import load_dataset, Dataset
import re
from typing import Literal
import pandas as pd
import os


# Define the system prompt that instructs the model to use a specific format
SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<think>
{thinking}
</think>
<answer>
{answer}
</answer>
"""

# Prepare dataset
# Helper functions to extract answers from different formats
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    answer = answer.replace(",", "")
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Function to prepare the GSM8K dataset
def get_gsm8k_questions(split: Literal["train", "test"]) -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    data = data.filter(lambda x: x["answer"] is not None)
    
    # Remove question column
    data = data.remove_columns("question")
    return data

def messages_to_text(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return {"text": text}
    
# Reward functions
# Reward function that checks if the answer is correct
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# Reward function that checks if the answer is an integer
def int_reward_func(completions, **_) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.lstrip('-').isdigit() else 0.0 for r in extracted_responses]

# Reward function that checks if the completion follows the strict format
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Reward function that checks if the completion follows a more relaxed format
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Reward function that counts XML tags and penalizes extra content
def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Function for saving results
def save_predictions_to_parquet(dataset, predictions, output_path):

    question = [i["prompt"][-1]["content"] for i in dataset]
    gt_answers = dataset["answer"]
    predicted_answers = [extract_xml_answer(pred.outputs[0].text) for pred in predictions]
    finish_reason = [pred.outputs[0].finish_reason for pred in predictions]

    df = pd.DataFrame({
        "question": question,
        "true_answer": gt_answers,
        "predicted_answer": predicted_answers,
        "finish_reason": finish_reason
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    return df