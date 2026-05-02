import numpy as np

robust    = np.load("robust_adapter.npz")
unlearned = np.load("elite_unlearned_adapter.npz")
oracle    = np.load("oracle_adapter.npz")

dist_robust_to_oracle    = np.sum([np.linalg.norm(robust[f]    - oracle[f]) for f in robust.files])
dist_unlearned_to_oracle = np.sum([np.linalg.norm(unlearned[f] - oracle[f]) for f in unlearned.files])

print("=== ORACLE RETRAIN COMPARISON ===")
print(f"Distance (Robust   -> Oracle) : {dist_robust_to_oracle:.4f}")
print(f"Distance (Unlearned -> Oracle) : {dist_unlearned_to_oracle:.4f}")

if dist_unlearned_to_oracle < dist_robust_to_oracle:
    print("\nResult: Unlearned model moved closer to the Oracle baseline.")
else:
    print("\nResult: Task Arithmetic did not move closer to the Oracle baseline.")
