import torch
import numpy as np
from datasets import load_dataset
from llm_model import setup_model, set_lora_weights, get_lora_weights

print("Testing global model on Swiss institutional data...")
model, tokenizer = setup_model()

# Load the robust adapter generated from the federated run.
data = np.load("robust_adapter.npz")
weights = [data[f] for f in data.files]
_, keys = get_lora_weights(model)
set_lora_weights(model, weights, keys)

swiss_data = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train[400:420]")

model.eval()

for i, example in enumerate(swiss_data):
    prompt = f"Question: {example['input']} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[Test {i+1}] {example['input'][:50]}...")
    print(f"[Model]: {response.split('Answer:')[-1].strip()}")

print("\nInstitutional generalization evaluation complete.")
