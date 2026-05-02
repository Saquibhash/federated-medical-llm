import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

alphas    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
forget    = [2.9972, 2.9986, 3.0000, 3.0014, 3.0027, 3.0040, 3.0053, 3.0072, 3.0102]
retain    = [5.5371, 5.5597, 5.5823, 5.6047, 5.6270, 5.6491, 5.6709, 5.7032, 5.7554]
nonmember = 3.1225

forget_delta = [f - forget[0] for f in forget]
retain_delta = [r - retain[0] for r in retain]

fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(alphas, forget, 'o-', color='#E24B4A', linewidth=2,
         markersize=6, label='Forget set — Hungary')
ax1.plot(alphas, retain, 's-', color='#1D9E75', linewidth=2,
         markersize=6, label='Retain set — Cleveland')
ax1.axhline(y=nonmember, color='#888780', linestyle='--',
            linewidth=1.5, label=f'Non-member baseline ({nonmember:.4f})')
ax1.set_xlabel('Alpha (unlearning strength)', fontsize=12)
ax1.set_ylabel('Mean NLL Loss', fontsize=12)
ax1.set_title('Panel A: Absolute NLL loss per alpha', fontsize=12, fontweight='normal')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25, linestyle='--')

ax1.annotate('alpha = 1.0\n(selected setting)',
             xy=(1.0, forget[alphas.index(1.0)]),
             xytext=(1.2, 3.08),
             arrowprops=dict(arrowstyle='->', color='#E24B4A', lw=1.2),
             fontsize=8, color='#E24B4A')

ax2 = fig.add_subplot(gs[1])
ax2.plot(alphas, forget_delta, 'o-', color='#E24B4A', linewidth=2,
         markersize=6, label='Forget set delta')
ax2.plot(alphas, retain_delta, 's-', color='#1D9E75', linewidth=2,
         markersize=6, label='Retain set delta')
ax2.axhline(y=0, color='#888780', linestyle='--', linewidth=1.2, label='No change')
ax2.set_xlabel('Alpha (unlearning strength)', fontsize=12)
ax2.set_ylabel('ΔNLL (relative to alpha=0 baseline)', fontsize=12)
ax2.set_title('Panel B: Loss change relative to baseline\n(proportional scaling, limited target specificity)',
              fontsize=12, fontweight='normal')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25, linestyle='--')

ax2.annotate('Both sets rise proportionally\n— limited target specificity',
             xy=(1.5, retain_delta[alphas.index(1.5)]),
             xytext=(0.5, 0.15),
             arrowprops=dict(arrowstyle='->', color='#444', lw=1.0),
             fontsize=8, color='#444')

plt.suptitle('Task Arithmetic alpha sweep: FedAvg approximation analysis',
             fontsize=13, fontweight='normal', y=1.02)

plt.tight_layout()
plt.savefig('alpha_sweep.png', dpi=150, bbox_inches='tight')
print("Saved: alpha_sweep.png")

print("\n" + "=" * 65)
print("  ALPHA SWEEP SUMMARY")
print("=" * 65)
print(f"\n  Forget loss range  : {min(forget):.4f} → {max(forget):.4f}  (Δ = {max(forget)-min(forget):.4f})")
print(f"  Retain loss range  : {min(retain):.4f} → {max(retain):.4f}  (Δ = {max(retain)-min(retain):.4f})")
print(f"  Non-member baseline: {nonmember:.4f}")
print(f"\n  {'Alpha':<6}  {'F/R ratio':<12}  {'Forget delta':>12}  {'Retain delta':>12}")
print("  " + "-" * 46)
for i, a in enumerate(alphas):
    print(f"  {a:<6.1f}  {forget[i]/retain[i]:<12.4f}  {forget_delta[i]:>+12.4f}  {retain_delta[i]:>+12.4f}")
