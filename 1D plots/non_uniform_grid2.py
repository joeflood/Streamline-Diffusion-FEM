import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

# Parameters
L = 1.0          # Domain length
N = 10           # Number of subintervals (increased for better resolution)
beta = 3         # Transformation parameter (beta > 1 clusters points near x=1)
xi = np.linspace(0, 1, N+1)
x = 1 - (1 - xi)**beta  # Nonuniform grid with a boundary layer towards x=1
h_ref = L / N  # Reference step size for computing nu

# Peclet numbers and convection velocity
Pe_values = [1.0, 5.0, 13.0]
c = 1.0  # Convection velocity

# Create a single figure with two rows (first for solution plots, second for Peclet distribution)
fig = plt.figure(figsize=(9, 4))

# Define grid layout: 2 rows and 1 column
gs = fig.add_gridspec(2, 3, height_ratios=[3, 2])  # First row takes more space
axes1 = [fig.add_subplot(gs[0, i]) for i in range(3)]  # Subplots for solution plots (first row)
axes2 = [fig.add_subplot(gs[1, i]) for i in range(3)]  # Subplots for Peclet number distribution (second row)

for idx, Pe in enumerate(Pe_values):
    # Use the reference spacing to set nu
    nu = c * h_ref / (2 * Pe)

    # Initialize system matrix and RHS vector
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    peclet = np.zeros(N)

    # Apply Dirichlet boundary conditions
    A[0, 0] = 1
    b[0] = 0  # u(0) = 0
    A[-1, -1] = 1
    b[-1] = 1  # u(1) = 1

    # Fill in the matrix for interior nodes
    for i in range(1, N):
        dx_w = x[i] - x[i-1]
        dx_e = x[i+1] - x[i]
        
        # Diffusion term coefficients
        aW_diff = nu * 2 / (dx_w * (dx_w + dx_e))
        aE_diff = nu * 2 / (dx_e * (dx_w + dx_e))
        aP_diff = -(aW_diff + aE_diff)
        
        # Convection term coefficients
        aW_conv = c / (2 * (dx_w + dx_e))
        aE_conv = - c / (2 * (dx_w + dx_e))
        
        # Assemble matrix
        A[i, i-1] = aW_diff + aW_conv
        A[i, i]   = aP_diff
        A[i, i+1] = aE_diff + aE_conv
        
        peclet[i] = c * dx_w / (2 * nu)

    # Solve the linear system
    u_numerical = np.linalg.solve(A, b)

    # Analytical solution
    PeG = c * L / (2 * nu)
    u_analytical = (1 - np.exp(PeG * x)) / (1 - np.exp(PeG))

    # --- First figure: Solution plots ---
    ax1 = axes1[idx]
    ax1.plot(x, u_numerical, '-', label="Numerical")
    ax1.plot(x, u_analytical, '--', linewidth = 0.75, label="Exact", color='k')
    ax1.plot(x, np.full(np.size(x), -0.085), '|', color='k', markersize=7, label="Grid Points")
    ax1.set_title(fr"$Pe_{{\text{{global}}}} = {Pe}$")
    if Pe == 0.5:
        ax1.set_ylabel('u(x)')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.1, 1.1)

    ax1.tick_params(axis='x', which='both', bottom=False)
    ax1.tick_params(axis='y', which='minor', left=False)
    # Turn off ticks for the top and right axes (both major and minor)
    ax1.tick_params(axis='y', which='both', right=False)
    ax1.tick_params(axis='x', which='both', top=False)



    # --- Second figure: Peclet number distribution ---
    ax2 = axes2[idx]
    ax2.plot(np.linspace(0, 1, N), peclet, 'b-')
    ax2.set_xlabel('x')
    if Pe == 0.5:
        ax2.set_ylabel('Local Pe')
    ax2.axhline(y=1.37, color='k', linestyle='--', linewidth = 0.75, label="Pe = 1.37")
    ax2.tick_params(axis='x', which='both', bottom=False)
    ax2.tick_params(axis='y', which='minor', left=False)
    # Turn off ticks for the top and right axes (both major and minor)
    ax2.tick_params(axis='y', which='both', right=False)
    ax2.tick_params(axis='x', which='both', top=False)
    ax2.margins(y=0)  # This removes margins for the y-axis (bottom and top)
    ax2.set_ylim(bottom=ax2.get_ylim()[0], top=ax2.get_ylim()[1])  # Keeps the top margin

# Adjust layout and display the figure
plt.tight_layout()
plt.show()
