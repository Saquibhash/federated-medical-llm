import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from llm_model import setup_model, set_lora_weights, get_lora_weights

model, tokenizer = setup_model()
_, keys = get_lora_weights(model)

unlearned_data = np.load("elite_unlearned_adapter.npz")
set_lora_weights(model, [unlearned_data[f] for f in unlearned_data.files], keys)

def get_sequence_losses(model, tokenizer, slice_str, n_samples=50):
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split=slice_str)
    losses = []
    model.eval()
    with torch.no_grad():
        for ex in list(ds)[:n_samples]:
            text = f"Q: {ex['input']} A: {ex['output']}"
            inputs = tokenizer(text, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            losses.append(outputs.loss.item())
    return np.array(losses)

print("Calculating losses for visualization (this will take a minute)...")
forget_losses = get_sequence_losses(model, tokenizer, "train[200:250]")
retain_losses = get_sequence_losses(model, tokenizer, "train[0:50]")
nonmember_losses = get_sequence_losses(model, tokenizer, "train[500:550]")

plt.figure(figsize=(9, 6))
sns.kdeplot(retain_losses, fill=True, color="#1D9E75", label="Retain Set (Cleveland)", alpha=0.5)
sns.kdeplot(forget_losses, fill=True, color="#E24B4A", label="Forget Set (Hungary)", alpha=0.5)
sns.kdeplot(nonmember_losses, fill=True, color="#888780", linestyle="--", label="Non-Member Baseline", alpha=0.3)

plt.title("Sequence-Level Loss Distribution Post-Unlearning", fontsize=14, pad=15)
plt.xlabel("Negative Log-Likelihood (NLL) Loss", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("mia_distribution.png", dpi=150)
print("Saved to mia_distribution.png")