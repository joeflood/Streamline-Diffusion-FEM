import matplotlib.pyplot as plt

# Data
Pe = [1, 2.5, 5, 10, 20, 50, 100, 200]
galerkin     = [5.24326, 9.74016, 16.21868, 25.39173, 38.19257, 68.53897, 118.59616, 220.63704]
supg_asym    = [5.24326, 7.69315, 11.28647, 16.28448, 23.25083, 36.95271, 52.33722, 74.06173]
supg_critical= [4.90291, 7.66768, 11.29689, 16.29261, 23.25680, 36.95585, 52.33902, 74.06267]

epsilon2 = [1/24, 1/36, 1/48, 1/60, 1/72, 1/96, 1/120]
Pe2 = [1, 1.5, 2, 2.5, 2.88, 4, 5]
galerkin2      = [5.24326, 6.64895, 8.20459, 9.74016, 11.19892, 13.85746, 16.21868]
supg_asym2     = [5.24326, 5.90933, 6.80724, 7.69315, 8.51899, 9.99368, 11.28647]
supg_critical2 = [4.90291, 5.80489, 6.75385, 7.66768, 8.53102, 10.00538, 11.29689]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: full range
ax1.plot(Pe, galerkin, label="Galerkin", marker='o')
ax1.plot(Pe, supg_asym, label="SUPG Asymptotic", marker='s')
ax1.plot(Pe, supg_critical, label="SUPG Critical", marker='^')
ax1.set_xlabel("Péclet Number (Pe)")
ax1.set_ylabel("L2 Gradient Error")
ax1.set_title("Full Range")
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend()

# Right plot: zoom on first 4 values


ax2.plot(Pe2, galerkin2, label="Galerkin", marker='o')
ax2.plot(Pe2, supg_asym2, label="SUPG Asymptotic", marker='s')
ax2.plot(Pe2, supg_critical2, label="SUPG Critical", marker='^')
ax2.set_xlabel("Péclet Number (Pe)")
ax2.set_title("Zoomed In (Pe ≤ 5)")
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
