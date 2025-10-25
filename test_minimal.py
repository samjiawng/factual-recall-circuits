import sys
sys.path.insert(0, 'src')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Paris is in", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
