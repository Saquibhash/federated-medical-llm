import numpy as np
import matplotlib.pyplot as plt

robust = np.load("robust_adapter.npz")
unlearned = np.load("elite_unlearned_adapter.npz")

a_drifts = []
b_drifts = []

for i in range(0, 12, 2):
    a_drifts.append(np.linalg.norm(robust[f'arr_{i}'] - unlearned[f'arr_{i}']))
    b_drifts.append(np.linalg.norm(robust[f'arr_{i+1}'] - unlearned[f'arr_{i+1}']))

layers = [f"Layer {i+1}" for i in range(6)]
x = np.arange(len(layers))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, a_drifts, width, label='A-Matrix Drift (Down-projection)', color='#2B5B84')
ax.bar(x + width/2, b_drifts, width, label='B-Matrix Drift (Up-projection)', color='#E24B4A')

ax.set_ylabel('L2 Parameter Drift', fontsize=12)
ax.set_title('Rank-Invariant LoRA Asymmetry During Task Arithmetic Unlearning', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('lora_asymmetry_chart.png', dpi=150)
print("Saved to lora_asymmetry_chart.png")