import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0     # Domain length
h = 0.1     # Step size
N = int(L / h)  # Number of interior points
x = np.linspace(0, L, N+1)  # Grid points

# Peclet numbers
Pe_values = [0.5, 1, 2.5]
c = 3.0  # Convection velocity

fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Create subplots

for idx, Pe in enumerate(Pe_values):
    nu = c * h / (2 * Pe)  # Compute diffusion coefficient
    tau = h / (2 * np.abs(c))  # Streamline diffusion stabilization parameter

    # Coefficients with streamline diffusion term
    aW = nu / h**2 + c / (2*h) + tau * c**2 / h**2
    aP = -2 * nu / h**2 + tau * (-2 * c**2 / h**2)
    aE = nu / h**2 - c / (2*h) + tau * c**2 / h**2

    # System matrix and RHS vector
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # Apply boundary conditions
    A[0, 0] = 1
    b[0] = 0  # u(0) = 0

    A[-1, -1] = 1
    b[-1] = 1  # u(1) = 1

    # Fill in the matrix for interior nodes
    for i in range(1, N):
        A[i, i-1] = aW
        A[i, i] = aP
        A[i, i+1] = aE
        b[i] = 1  # No source term

    # Solve the system
    u_numerical = np.linalg.solve(A, b)

    # Analytical solution
    PeG = c * L / nu
    u_analytical = x/c + (1-1/c)*(np.exp(c/nu*x)-1)/(np.exp(c/nu)-1)

    # Plot results in subplot
    ax = axes[idx]
    ax.plot(x, u_numerical, '-', label=f"Numerical (SD)")
    ax.plot(x, u_analytical, '--', label=f"Exact")
    ax.set_title(f"Pe = {Pe}")
    ax.legend()
    ax.set_xticks(np.linspace(0, 1, 3))  
    ax.set_yticks(np.linspace(-0.5, 1, 4))  
    ax.set_xlim(0, 1)  
    ax.set_ylim(-0.5, 1)  
    ax.margins(0)  
    ax.grid(True)

plt.suptitle("1D Convection-Diffusion with Streamline Diffusion")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
