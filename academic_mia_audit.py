"""
Sequence-Level MIA Audit — Negative Log-Likelihood (NLL) & ROC-AUC
===================================================================
Evaluates selective erasure using per-sequence NLL and the ROC-AUC
metric, providing empirical evidence of reduced membership signal
after unlearning. A score near 0.5 indicates the attacker is
statistically indistinguishable from random guessing.
"""

import torch
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from llm_model import setup_model, set_lora_weights, get_lora_weights
from datasets import load_dataset


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


print("=" * 65)
print("  SEQUENCE-LEVEL MIA AUDIT (NLL & ROC-AUC)")
print("=" * 65)

model, tokenizer = setup_model()
_, keys = get_lora_weights(model)

print("\n[1/4] Evaluating robust model (pre-unlearning baseline)...")
robust_data = np.load("robust_adapter.npz")
set_lora_weights(model, [robust_data[f] for f in robust_data.files], keys)

forget_losses_before = get_sequence_losses(model, tokenizer, "train[200:250]")
retain_losses_before = get_sequence_losses(model, tokenizer, "train[0:50]")

print(f"    Forget set (Hungary)   — mean loss: {np.mean(forget_losses_before):.4f}")
print(f"    Retain set (Cleveland) — mean loss: {np.mean(retain_losses_before):.4f}")

print("\n[2/4] Loading unlearned model (Task Arithmetic)...")
elite_data = np.load("elite_unlearned_adapter.npz")
set_lora_weights(model, [elite_data[f] for f in elite_data.files], keys)

w_drift = np.linalg.norm(robust_data[robust_data.files[0]] - elite_data[elite_data.files[0]])
print(f"    Weight drift on arr_0: {w_drift:.6f}")

print("\n[3/4] Evaluating unlearned model (post-unlearning)...")
forget_losses_after  = get_sequence_losses(model, tokenizer, "train[200:250]")
retain_losses_after  = get_sequence_losses(model, tokenizer, "train[0:50]")
nonmember_losses     = get_sequence_losses(model, tokenizer, "train[500:550]")

print(f"    Forget set (Hungary)   — mean loss: {np.mean(forget_losses_after):.4f}")
print(f"    Retain set (Cleveland) — mean loss: {np.mean(retain_losses_after):.4f}")
print(f"    Non-member (control)   — mean loss: {np.mean(nonmember_losses):.4f}")

print("\n[4/4] Computing statistical significance & ROC-AUC...\n")

t1, p1 = stats.ttest_ind(forget_losses_after, nonmember_losses)
pooled_std_1 = np.sqrt((np.std(forget_losses_after)**2 + np.std(nonmember_losses)**2) / 2)
cohen_d1 = (np.mean(forget_losses_after) - np.mean(nonmember_losses)) / (pooled_std_1 + 1e-9)

all_losses  = np.concatenate([forget_losses_after, nonmember_losses])
true_labels = np.concatenate([np.ones(len(forget_losses_after)), np.zeros(len(nonmember_losses))])
auc_score   = roc_auc_score(true_labels, -all_losses)

status = "no significant difference observed" if p1 > 0.05 else "significant difference observed"

print("  Test 1 — Forget set vs Non-member loss distribution:")
print(f"    T-Statistic : {t1:.4f}")
print(f"    P-Value     : {p1:.4f}  ({status})")
print(f"    Cohen's d   : {cohen_d1:.4f}  (|d| < 0.2 = negligible)")
print(f"    MIA ROC-AUC : {auc_score:.4f}  (near 0.5 = membership inference at chance level)")

t2, p2 = stats.ttest_rel(retain_losses_after, retain_losses_before)
retain_delta = np.mean(retain_losses_after) - np.mean(retain_losses_before)

print(f"\n  Test 2 — Retain set stability (Utility Preservation):")
print(f"    Mean loss change : {retain_delta:+.4f}")
print(f"    P-Value          : {p2:.4f}  {'STABLE' if p2 > 0.05 else 'UTILITY SHIFTED'}")

a_drifts = []
b_drifts = []

for arr_name, key in zip(robust_data.files, keys):
    drift = np.linalg.norm(robust_data[arr_name] - elite_data[arr_name])
    if "lora_A" in key:
        a_drifts.append(drift)
    elif "lora_B" in key:
        b_drifts.append(drift)

if np.mean(b_drifts) > 0:
    ratio = np.mean(a_drifts) / np.mean(b_drifts)
    print(f"\n  LoRA Asymmetry:")
    print(f"    Mean A-matrix drift : {np.mean(a_drifts):.6f}")
    print(f"    Mean B-matrix drift : {np.mean(b_drifts):.6f}")
    print(f"    Asymmetry ratio     : {ratio:.1f}x")

print("─" * 65)
