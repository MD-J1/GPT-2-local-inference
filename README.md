# GPT-2-local-inference
GPT-2 inference using Hugging Face Transformers in Python

---


This repository demonstrates running **GPT-2** using Hugging Face Transformers in Python. GPT-2 is small enough to run in **GitHub Codespaces** on the free plan without exceeding memory or CPU limits.

## Features

* **Deterministic output:** The code sets a fixed random seed for reproducible results.
* **Greedy decoding:** The model always picks the token with the highest probability (`do_sample=False`), avoiding random sampling or stochastic outputs.
* **Customizable output length:** Users can adjust the `max_new_tokens` parameter in `A.py` to generate longer or shorter responses.

## Usage

1. Install dependencies:

```bash
pip install torch transformers
```

2. Run the script:

```bash
python A.py
```

3. Modify `prompt` and `max_new_tokens` in the script to test different inputs and output lengths.

---

