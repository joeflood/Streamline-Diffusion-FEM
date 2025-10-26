import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 1.0  # physical diffusion coefficient
Pe = np.linspace(0, 2.5, 250)  # range of Peclet numbers from 0 to 3

# Effective diffusion coefficient
nu_eff = nu * (1 - (Pe**2)/3)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(Pe, nu_eff, label=r'$\nu_{\rm eff} = \nu\left(1-\frac{Pe^2}{3}\right)$', lw=2)
plt.axhline(0, color='k', linestyle='--', lw=1)  # horizontal line at nu_eff = 0
plt.xlabel('Peclet Number (Pe)', fontsize=12)
plt.ylabel('Effective Diffusion Coefficient', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
