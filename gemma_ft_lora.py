"""## Installation"""

# pip install transformers==4.38.2
# pip install -q  torch peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7 accelerate

"""# Prepare Data

"""

# import random
# import json
# from transformers import AutoTokenizer

# # Define operations
# operations = ['+', '-', 'x', '/']

# base_model_name = "google/gemma-2b-it"

# # Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# # Function to generate a random arithmetic expression
# def generate_expression(tokenizer):
#     a = random.randint(0, 1000)
#     b = random.randint(0, 1000)
#     operation = random.choice(operations)

#     if operation == '+':
#         result = a + b
#     elif operation == '-':
#         result = a - b
#     elif operation == 'x':
#         result = a * b
#     elif operation == '/':
#         # Ensure the second operand is not zero and the division results in a reasonable float
#         while b == 0 or a % b != 0:
#             b = random.randint(1, 100)
#         result = a // b  # Use integer division for simplicity

#     return tokenizer.apply_chat_template([{"role": "user", "content": f"{a} {operation} {b} ="}, {"role": "assistant", "content": result}], tokenize=False)

# # Generate 100 expressions
# training_data = [generate_expression(tokenizer) for _ in range(3000)]

# # Save dataset as JSONL file
# with open('arithmetic_dataset.jsonl', 'w') as f:
#     for entry in training_data:
#         f.write(json.dumps(entry) + '\n')

# print("Dataset saved as 'arithmetic_dataset.jsonl'.")

"""## Upload to Hf"""

# from datasets import Dataset, DatasetDict

# # Load the jsonl file
# data = []
# with open('arithmetic_dataset.jsonl', 'r') as f:
#     for line in f:
#         data.append(json.loads(line))

# # Create Hugging Face Dataset
# hf_dataset = Dataset.from_dict({
#     "text": [entry for entry in data]
# })

# # Save the dataset locally as preparation to push to HF
# hf_dataset.save_to_disk("hf_arithmetic_dataset")

# # Load and push to Hugging Face (replace 'your-username' and 'dataset-name' with appropriate values)
# hf_dataset.push_to_hub("Harsh1729/simple_arithmetic")

"""# Load and Train"""

# !pip install -q  torch peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7 accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

# Dataset
data_name = "Harsh1729/simple_arithmetic"
training_data = load_dataset(data_name, split="train")

# Model and tokenizer names
base_model_name = "google/gemma-2b-it"
refined_model = "gemma-2b-it-arithmetic" #You can give it your own name

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Target Modules
target_modules=[
"q_proj",
"k_proj",
"v_proj",
"o_proj",
"gate_proj",
"up_proj",
"down_proj",
"lm_head",
]

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
    push_to_hub=True,
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)

# Generate Text
query = "55 - 8 ="
prompt = tokenizer.apply_chat_template([{"role": "user", "content": query}], tokenize=False)
text_gen = pipeline(task="text-generation", model=refined_model, tokenizer=tokenizer, max_length=200)
output = text_gen(prompt)
print(output[0]['generated_text'])