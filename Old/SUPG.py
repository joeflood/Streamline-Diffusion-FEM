import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Parameters
nx, ny = 10, 10   # Number of elements in x and y directions
Lx, Ly = 1.0, 1.0 # Domain size
dx, dy = Lx / nx, Ly / ny  # Element sizes
gamma = 1  # Diffusion coefficient
v = np.array([10, 10])  # Convection velocity [vx, vy]
tau = 0.1 * dx  # SUPG stabilization parameter

# Generate mesh
x = np.linspace(0, Lx, nx+1)
y = np.linspace(0, Ly, ny+1)
X, Y = np.meshgrid(x, y)

# Number of nodes
num_nodes = (nx+1) * (ny+1)

# Define the source function
f = np.zeros(num_nodes)

# Define boundary conditions (Dirichlet)
def u_exact(x, y):
    return 1.0 + x**2 + 2*y**2  # Example exact solution

# Global stiffness matrix and load vector
K = lil_matrix((num_nodes, num_nodes))
F = np.zeros(num_nodes)

# Map from 2D (i,j) to global node number
def node_num(i, j):
    return j * (nx + 1) + i

# Assembly loop
for i in range(nx):
    for j in range(ny):
        # Get global node numbers of the element
        nodes = [node_num(i, j), node_num(i+1, j), node_num(i+1, j+1), node_num(i, j+1)]
        xc, yc = x[i] + dx/2, y[j] + dy/2  # Element center

        # Standard bilinear shape function gradients
        grad_N = np.array([
            [-1/dx, -1/dy], [1/dx, -1/dy],
            [1/dx, 1/dy], [-1/dx, 1/dy]
        ])

        # Compute element stiffness matrix with SUPG stabilization
        A_elem = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                diffusive_term = gamma * np.dot(grad_N[a], grad_N[b]) * dx * dy
                convective_term = (np.dot(v, grad_N[b]) * (dx * dy) / 4)
                stabilization = tau * np.dot(v, grad_N[a]) * np.dot(v, grad_N[b]) * dx * dy
                A_elem[a, b] = diffusive_term + convective_term + stabilization

        # Assemble into global system
        for a in range(4):
            for b in range(4):
                K[nodes[a], nodes[b]] += A_elem[a, b]
            F[nodes[a]] += f[nodes[a]] * dx * dy / 4  # Load vector

# Apply Dirichlet boundary conditions
for i in range(num_nodes):
    xi, yi = X.ravel()[i], Y.ravel()[i]  # Get x, y coordinates of the node
    if xi == 0 or yi == Ly:  # Left boundary (x = 0)
        K[i, :] = 0    # Zero out the row
        K[i, i] = 1    # Set diagonal to 1
        F[i] = 1      # Set value to 1
    elif xi == Lx or yi == 0:  # Other boundaries (x = Lx, y = 0, y = Ly)
        K[i, :] = 0    # Zero out the row
        K[i, i] = 1    # Set diagonal to 1
        F[i] = 0       # Set value to 0


# Convert to sparse format
K = csr_matrix(K)

# Solve the linear system
u = spsolve(K, F)

# Reshape for plotting
u_grid = u.reshape((ny+1, nx+1))

# Plot the solution
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u_grid, 50, cmap='viridis')
plt.colorbar(label="Solution u")
plt.xlabel("x")
plt.ylabel("y")
plt.title("SUPG-Stabilized Convection-Diffusion Solution")
plt.show()

# 3D Trisurf Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X.ravel(), Y.ravel(), u, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("3D Trisurf Plot")
ax.view_init(elev=25, azim=145)
plt.show()