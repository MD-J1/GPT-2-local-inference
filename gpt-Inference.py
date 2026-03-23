import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model + tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # inference mode

# fixed random seed for deterministic results
torch.manual_seed(42)

prompt = "Hi chat who is the current president of the US"
inputs = tokenizer(prompt, return_tensors="pt")

# generate deterministic output
generated = model.generate(
    **inputs,
    max_new_tokens=67,
    do_sample=False,   # greedy = always pick top token
)

# decode and print
output_text = tokenizer.decode(generated[0])
print(output_text)