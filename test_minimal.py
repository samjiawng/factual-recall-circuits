import sys
sys.path.insert(0, 'src')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Testing basic inference...")
inputs = tokenizer("Paris is in", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print(f"✓ Output shape: {outputs.logits.shape}")
print("Model works! The issue is in the circuit discovery code.")
