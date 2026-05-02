import torch
import numpy as np
from llm_model import setup_model, get_lora_weights, set_lora_weights

print("--- TASK ARITHMETIC UNLEARNING ---\n")

model, tokenizer = setup_model()
_, keys = get_lora_weights(model)

robust = np.load("robust_adapter.npz")
robust_weights = [robust[f] for f in robust.files]

N     = 10
alpha = 1.0

print(f"Nodes in federation : {N}")
print(f"Alpha               : {alpha}")
print(f"Hungary weight share: 1/{N} = {1/N:.2f}\n")

unlearned_weights = []
for r_weight in robust_weights:
    hungary_task_vector = r_weight / N
    unlearned_weights.append(r_weight - alpha * hungary_task_vector)

set_lora_weights(model, unlearned_weights, keys)
np.savez("elite_unlearned_adapter.npz", *unlearned_weights)

print("=== Weight Drift Report ===")
for f, key, r, u in zip(robust.files, keys, robust_weights, unlearned_weights):
    drift      = np.linalg.norm(r - u)
    layer_type = "A-matrix" if "lora_A" in key else "B-matrix"
    print(f"  {f} ({layer_type}): L2 drift = {drift:.4f}")

print("\nTask Arithmetic complete. elite_unlearned_adapter.npz saved.")
