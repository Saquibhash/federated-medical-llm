import matplotlib.pyplot as plt

rounds   = [1, 2, 3]
val_loss = [3.9463, 3.8820, 3.7717]

plt.figure(figsize=(8, 5))
plt.plot(rounds, val_loss, marker='o', linestyle='-', color='#2B5B84', linewidth=2.5, markersize=8)

plt.title('Federated Training Convergence (10 Nodes + Trimmed Mean)', fontsize=14, pad=15)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Aggregated Validation Loss (NLL)', fontsize=12)
plt.xticks(rounds)
plt.grid(True, linestyle='--', alpha=0.5)

for i, loss in enumerate(val_loss):
    plt.text(rounds[i], loss + 0.005, f'{loss:.4f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig('federated_convergence.png', dpi=150, bbox_inches='tight')
print("Saved: federated_convergence.png")
