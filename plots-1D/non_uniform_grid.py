import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

# Parameters
L = 1.0          # Domain length
N = 100          # Number of subintervals (increased for better resolution)
beta = 3         # Transformation parameter; beta>1 clusters points near x=1
xi = np.linspace(0, 1, N+1)
x = 1 - (1 - xi)**beta  # Nonuniform grid with boundary layer towards x=1

h_ref = L / N  # Reference step size for computing nu

# Peclet numbers and convection velocity
Pe_values = [0.05, 0.1, 0.25]
c = 1.0  # Convection velocity

fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # Create subplots

for idx, Pe in enumerate(Pe_values):
    # Use the reference spacing to set nu (this maintains a similar Pe scaling)
    nu = c * h_ref / (2 * Pe)
    
    # Initialize system matrix and RHS vector
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # Apply Dirichlet boundary conditions
    A[0, 0] = 1
    b[0] = 0   # u(0) = 0

    A[-1, -1] = 1
    b[-1] = 1  # u(1) = 1

    # Fill in the matrix for interior nodes (i = 1, ..., N-1)
    for i in range(1, N):
        dx_w = x[i] - x[i-1]
        dx_e = x[i+1] - x[i]
        
        # Diffusion term coefficients using nonuniform grid second derivative approximation
        aW_diff = nu * 2 / (dx_w * (dx_w + dx_e))
        aE_diff = nu * 2 / (dx_e * (dx_w + dx_e))
        aP_diff = -(aW_diff + aE_diff)
        
        # Convection term coefficients using central difference
        aW_conv = - c / (2 * (dx_w + dx_e))
        aE_conv = + c / (2 * (dx_w + dx_e))
        
        # Total coefficients
        A[i, i-1] = aW_diff + aW_conv
        A[i, i]   = aP_diff
        A[i, i+1] = aE_diff + aE_conv
        
        b[i] = 0  # No source term

    # Solve the system of equations
    u_numerical = np.linalg.solve(A, b)

    # Analytical solution (computed in the physical coordinate x)
    PeG = c * L / nu
    u_analytical = (1 - np.exp(PeG * x)) / (1 - np.exp(PeG))

    # Plot results in subplot
    ax = axes[idx]
    ax.plot(x, u_numerical, '-', label="Numerical")
    ax.plot(x, u_analytical, '--', label="Exact", color='k')
    ax.set_title(f"Pe = {Pe}")
    if Pe == 0.5:
        ax.set_ylabel('u(x)')
    ax.set_xlabel('x')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1)
    ax.margins(0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
