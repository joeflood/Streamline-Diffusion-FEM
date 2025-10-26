import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import scienceplots

plt.style.use('science')

# Parameters
nx, ny = 20, 20           # Number of elements in x and y directions
Lx, Ly = 1.0, 1.0         # Domain size
dx, dy = Lx / nx, Ly / ny  # Element sizes
gamma = 1/200             # Diffusion coefficient
v = np.array([1, 1])      # Convection velocity [vx, vy]
tau = dx / 2              # SUPG stabilization parameter

# Create uniform mesh
x = np.linspace(0, Lx, nx + 1)
y = np.linspace(0, Ly, ny + 1)
X, Y = np.meshgrid(x, y)
num_nodes = (nx + 1) * (ny + 1)

# Define source function (here f=0 everywhere)
f = np.zeros(num_nodes)

# Global stiffness matrix and load vector
K = lil_matrix((num_nodes, num_nodes))
F = np.zeros(num_nodes)

# Map from 2D (i, j) indices to a global node number
def node_num(i, j):
    return j * (nx + 1) + i

# Standard bilinear shape function gradients (constant per element)
grad_N = np.array([
    [-1/dx, -1/dy],
    [ 1/dx, -1/dy],
    [ 1/dx,  1/dy],
    [-1/dx,  1/dy]
])

# Assembly loop over elements
for i in range(nx):
    for j in range(ny):
        # Get global node numbers for the current element (quadrilateral)
        nodes = [node_num(i, j), node_num(i+1, j), node_num(i+1, j+1), node_num(i, j+1)]
        # Compute the element stiffness matrix including SUPG stabilization
        A_elem = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                diffusive_term   = gamma * np.dot(grad_N[a], grad_N[b]) * dx * dy
                convective_term  = np.dot(v, grad_N[b]) * (dx * dy) / 4
                stabilization    = tau * np.dot(v, grad_N[a]) * np.dot(v, grad_N[b]) * dx * dy
                A_elem[a, b] = diffusive_term + convective_term + stabilization

        # Assemble the local element matrix into the global system
        for a in range(4):
            for b in range(4):
                K[nodes[a], nodes[b]] += A_elem[a, b]
            F[nodes[a]] += f[nodes[a]] * dx * dy / 4

# Apply Dirichlet boundary conditions:
# Set u = 1 on the left boundary (x = 0) and u = 0 on the other boundaries.
def apply_boundary_conditions():
    global K, F, X, Y, num_nodes
    for i in range(num_nodes):
        xi, yi = X.ravel()[i], Y.ravel()[i]
        # Left boundary (x == 0)
        if np.isclose(xi, 0):
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 1
        # Other boundaries: right (x == Lx), bottom (y == 0), top (y == Ly)
        elif np.isclose(xi, Lx) or np.isclose(yi, 0) or np.isclose(yi, Ly):
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 0

apply_boundary_conditions()

# Convert stiffness matrix to CSR format and solve the linear system
K = csr_matrix(K)
u = spsolve(K, F)
u_grid = u.reshape((ny + 1, nx + 1))

# --------------------- Plotting ---------------------

# 1. 2D Plot using imshow
plt.figure(figsize=(8, 6))
plt.imshow(u_grid, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label="Solution u")
plt.xlabel("x")
plt.ylabel("y")
plt.title("SUPG-Stabilized Convection-Diffusion Solution (2D)")


# 2. 3D Surface Plot using quadrilateral elements
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
X_grid, Y_grid = np.meshgrid(x, y)
ax.plot_surface(X_grid, Y_grid, u_grid, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("3D Surface Plot with Quadrilateral Elements")

def plot_solution(ax, method, title):    
    num_contours = 31
    contours = np.linspace(0, 1, num_contours)
    contours = contours[1:num_contours-1]

    contour = ax.contour(X - 1, Y - 1, u_grid, levels=contours)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("u(x)")
    
    #ax.set_xticks(np.linspace(-1, 1, 5))
    #ax.set_yticks(np.linspace(-1, 1, 5))
    ax.tick_params(which='minor', bottom=False, top=False, left=False, right=False)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

# Create a 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Plot solutions
plot_solution(axes[0, 0], method="Galerkin", title="Galerkin")
plot_solution(axes[0, 1], method="Artificial", title="Artificial")
plot_solution(axes[1, 0], method="SUPG", title="SUPG")

# Leave the fourth subplot empty
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
