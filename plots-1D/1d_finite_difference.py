import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

# Parameters
L = 1.0     # Domain length
h = 0.1    # Step size
N = int(L / h)  # Number of interior points
x = np.linspace(0, L, N+1)  # Grid points

# Peclet numbers
Pe_values = [0.5, 1.0, 1.5]
c = 1.0  # Convection velocity

fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # Create subplots

for idx, Pe in enumerate(Pe_values):
    nu = c * h / (2 * Pe)  # Compute diffusion coefficient
    # Coefficients (adjusting convection term direction)
    aW = nu / h**2 + c / (2*h)
    aP = -2 * nu / h**2
    aE = nu / h**2 - c / (2*h)

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
        b[i] = 0  # No source term

    
    # Solve the system
    u_numerical = np.linalg.solve(A, b)

    # Analytical solution
    PeG = c * L / (nu)
    u_analytical = (1 - np.exp(PeG * x)) / (1 - np.exp(PeG))

    # Plot results in subplot
    ax = axes[idx]
    ax.plot(x, u_numerical, '-', label=f"Numerical")
    ax.plot(x, u_analytical, '--', label=f"Exact", color = 'k')
    ax.set_title(f"Pe = {Pe}")
    if Pe == 0.5:
        ax.set_ylabel('u(x)')
    
    ax.set_xlabel('x')
    ax.legend()
    #ax.set_xticks(np.linspace(0, 1, 3))  # Adjust number of ticks on x-axis
    #ax.set_yticks(np.linspace(-0.5, 1, 4))  # Adjust number of ticks on y-axis
    ax.set_xlim(0, 1)  # Ensure x-axis starts at 0 and ends at 1
    ax.set_ylim(-0.5, 1)  # Ensure y-axis starts at 0 and ends at 1

    ax.margins(0)  # Remove extra margins

    #ax.xaxis.set_major_locator(plt.MaxNLocator(2))  # Limits number of x-axis ticks
    #ax.yaxis.set_major_locator(plt.MaxNLocator(2))  # Limits number of y-axis ticks
    #ax.grid(True)

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
