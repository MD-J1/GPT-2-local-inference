import torch
from transformers import AutoModelForCausalLM

# Load the GPT-2 model
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # inference mode

# Dictionary to store all parameters
parameters = {}

# Loop over all model parameters
for name, param in model.named_parameters():
    parameters[name] = param.detach().clone()  # detach from computation graph
    print(f"{name}: {param.shape}")

# Now 'parameters' contains all weights and biases
# Example access:
# token embeddings: parameters['transformer.wte']
# positional embeddings: parameters['transformer.wpe']
# first attention weights: parameters['transformer.h.0.attn.c_attn.weight']