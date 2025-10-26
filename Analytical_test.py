import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Define the domain
Nx, Ny = 50, 50  # Grid resolution
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)

# Define parameters
epsilon = 0.01  # Small diffusion coefficient

def xi_0(eta):
    """Function to match the inflow data."""
    return np.where(eta <= 0.5, eta, eta + 0.1)  # Assuming Delta = 0.1

# Compute solution
U = 0.5 * (1 - erf((X + Y - 2 * xi_0((2 * Y - X) / 2)) / (2 * np.sqrt(epsilon))))

# Plot the solution
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, U, levels=50, cmap="jet")
plt.colorbar(contour, label="u(x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact Solution to 2D Steady Convection-Diffusion Equation")
plt.show()
