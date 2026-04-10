import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load the Model and Tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Prepare your Prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# 3. Get the Raw Logits (The "Truth")
with torch.no_grad():
    outputs = model(**inputs)
    # DistilGPT2 outputs logits for every token in the input.
    # We only want the logits for the VERY LAST token (the prediction for the next word).
    last_token_logits = outputs.logits[0, -1, :]

# 4. Find the Top 10 Candidates
top_k = 10
probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
top_values, top_indices = torch.topk(probs, top_k)

print(f"--- Top {top_k} Predictions for: '{prompt}' ---")
for i in range(top_k):
    token_id = top_indices[i].item()
    token_str = tokenizer.decode([token_id])
    probability = top_values[i].item() * 100
    print(f"Rank {i+1}: ID {token_id:5} | Token: '{token_str}' | Probability: {probability:.2f}%")