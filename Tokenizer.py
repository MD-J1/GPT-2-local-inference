import torch
from transformers import AutoTokenizer

# Load the GPT-2 tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Your prompt
prompt = "Hello world"

# Convert prompt to token IDs
inputs = tokenizer(prompt, return_tensors="pt")

# This gives you a tensor of token IDs
token_ids = inputs["input_ids"]
print("Token IDs:", token_ids)