import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(0, 1, 100)

# Define linearly independent linear functions
phi_1 = (x)                      # Identity function
phi_2 = (x)**2                   # Shifted linear function
phi_3 = (x)**3                 # Scaled and shifted linear function

# Plot the functions
plt.figure(figsize=(8, 6))

plt.plot(x, phi_1, label=r'$\phi_1(x) = x$')
plt.plot(x, phi_2, label=r'$\phi_2(x) = x^2$')
plt.plot(x, phi_3, label=r'$\phi_3(x) = x^3$')

# Formatting the plot
plt.xlabel('x')
plt.ylabel('Function Value')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid()

# Show the plot
plt.show()
