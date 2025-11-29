import torchvision
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from peft import LoraConfig
import re

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
train_dataset = load_dataset(dataset_id, split='train[:5%]')



model_name = "Qwen/Qwen3-VL-4B-Instruct" # "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
    "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)


def make_conversation(example):
    conversation = [
        {
            "role": "system",
            # "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": example["problem"],
            # "content": [
            #     {"type": "image", "image": example["image"]},
            #     {"type": "text", "text": example["problem"]},
            # ],
        },
    ]
    # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    prompt = conversation
    return {
        "prompt": prompt,
        # "prompt": prompt,
        "image": example["image"],
    }


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    breakpoint()
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [1.0 if match else 0.0 for match in matches]



def len_reward(completions, solution, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = completions

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards




if __name__ == "__main__":
    train_dataset = train_dataset.map(make_conversation)

    train_dataset = train_dataset.remove_columns(['problem', 'original_question', 'original_answer'])


    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype="auto",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ),
    )


    # You may need to update `target_modules` depending on the architecture of your chosen model.
    # For example, different VLMs might have different attention/projection layer names.
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    output_dir = "Qwen3-VL-4B-Instruct-trl-grpo"

    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        learning_rate=2e-5,
        #num_train_epochs=1,
        max_steps=100,                                        # Number of dataset passes. For full trainings, use `num_train_epochs` instead

        # Parameters that control the data preprocessing
        per_device_train_batch_size=2,
        max_completion_length=1024, # default: 256            # Max completion length produced during training
        num_generations=2, # 2, # default: 8                  # Number of generations produced during trainig for comparison
        max_prompt_length=2048, # default: 512                # Max prompt lenght of the input prompt used for generation during training

        fp16=True,

        # Parameters related to reporting and saving
        output_dir=output_dir,                                # Where to save model checkpoints and logs
        logging_steps=1,                                      # Log training metrics every N steps
        # report_to="wandb",                                  # Experiment tracking tool
        report_to="none",                                  # Experiment tracking tool
    )




    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, len_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer_stats = trainer.train()