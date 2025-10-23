import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Testing inference...")
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model(**inputs)
print("✓ Success! Model works.")
