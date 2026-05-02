import torch
import numpy as np
from datasets import load_dataset
from llm_model import setup_model, get_lora_weights, set_lora_weights

model, tokenizer = setup_model()


def get_mean_loss(model, tokenizer, slice_str, n_samples=30):
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split=slice_str)
    losses = []
    model.eval()
    with torch.no_grad():
        for ex in list(ds)[:n_samples]:
            text = f"Q: {ex['input']} A: {ex['output']}"
            inputs = tokenizer(
                text, padding="max_length", max_length=64,
                truncation=True, return_tensors="pt",
            )
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            losses.append(outputs.loss.item())
    return np.mean(losses)


robust = np.load("robust_adapter.npz")
robust_weights = [robust[f] for f in robust.files]
_, keys = get_lora_weights(model)

set_lora_weights(model, robust_weights, keys)
nonmember_baseline = get_mean_loss(model, tokenizer, "train[500:550]")

print("\n--- ALPHA SWEEP: TASK ARITHMETIC TRADE-OFF CURVE ---")
print(f"Non-member baseline: {nonmember_baseline:.4f}")
print(f"\n{'Alpha':<6} | {'Forget Loss':>12} | {'Retain Loss':>12} | {'Forget vs NM':>14} | {'Status'}")
print("-" * 72)

alphas  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
results = []

for alpha in alphas:
    if alpha == 0.0:
        set_lora_weights(model, robust_weights, keys)
    else:
        unlearned_weights = [w - alpha * (w / 10.0) for w in robust_weights]
        set_lora_weights(model, unlearned_weights, keys)

    f_loss = get_mean_loss(model, tokenizer, "train[200:250]")
    r_loss = get_mean_loss(model, tokenizer, "train[0:50]")

    diff = f_loss - nonmember_baseline
    if alpha == 0.0:
        status = "BASELINE"
    elif abs(diff) < 0.05:
        status = "*** OPTIMAL ***"
    elif diff > 0.3:
        status = "under-unlearned"
    elif diff < -0.1:
        status = "over-unlearned"
    else:
        status = "converging"

    results.append((alpha, f_loss, r_loss, diff))
    print(f"{alpha:<6.1f} | {f_loss:>12.4f} | {r_loss:>12.4f} | {diff:>+14.4f} | {status}")

print("-" * 72)
print("\nDone.")
