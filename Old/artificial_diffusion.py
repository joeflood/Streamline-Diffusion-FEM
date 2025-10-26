import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Parameters
nx, ny = 15, 15       # Number of elements in x and y directions
Lx, Ly = 1.0, 1.0     # Domain size
dx, dy = Lx / nx, Ly / ny  # Element sizes

# Diffusion coefficients and convection velocity
delta = 0.2        # Artificial diffusion coefficient
epsilon = 0.005        # Physical diffusion coefficient
w = np.array([0, -1])  # Convection velocity [w_x, w_y]

# Effective diffusion coefficient in the equation: (delta-epsilon)
effective_diff = delta - epsilon

# Generate mesh
x = np.linspace(-1, Lx, nx+1)
y = np.linspace(-1, Ly, ny+1)
X, Y = np.meshgrid(x, y)

# Number of nodes
num_nodes = (nx+1) * (ny+1)

# Define the source function f (homogeneous equation => f=0)
f = np.zeros(num_nodes)

# Global stiffness matrix and load vector
K = lil_matrix((num_nodes, num_nodes))
F = np.zeros(num_nodes)

# Map from 2D (i,j) to global node number
def node_num(i, j):
    return j * (nx + 1) + i

# Assembly loop
for i in range(nx):
    for j in range(ny):
        # Global node numbers for the current quadrilateral element
        nodes = [node_num(i, j), node_num(i+1, j), node_num(i+1, j+1), node_num(i, j+1)]
        
        # For a rectangular element with bilinear shape functions,
        # the gradients are constant over the element:
        grad_N = np.array([
            [-1/dx, -1/dy],
            [ 1/dx, -1/dy],
            [ 1/dx,  1/dy],
            [-1/dx,  1/dy]
        ])
        
        A_elem = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                # Diffusive term using the effective diffusion coefficient:
                diffusive_term = effective_diff * np.dot(grad_N[a], grad_N[b]) * dx * dy
                # Convection term:
                convective_term = np.dot(w, grad_N[b]) * (dx * dy) / 4
                # Sum the contributions (no SUPG stabilization term here)
                A_elem[a, b] = diffusive_term + convective_term
        
        # Assemble local element matrix into global stiffness matrix and load vector
        for a in range(4):
            for b in range(4):
                K[nodes[a], nodes[b]] += A_elem[a, b]
            F[nodes[a]] += f[nodes[a]] * dx * dy / 4  # here f is zero

# Apply Dirichlet boundary conditions.
# In this example, we set u = 1 on the left boundary (x=0) and u = 0 on all other boundaries.
for i in range(num_nodes):
    xi, yi = X.ravel()[i], Y.ravel()[i]
    # Apply Dirichlet condition u = 0 on the bottom boundary (y = 0)
    if yi == -1 or xi == -1 or xi == 1:
        K[i, :] = 0
        K[i, i] = 1
        F[i] = 0  # u = 0 at the bottom boundary
    
    # Apply Dirichlet condition u = 4 - (x - 1)^2 on the top boundary (y = L_y)
    elif yi == Ly:
        K[i, :] = 0
        K[i, i] = 1
        F[i] = 4-(xi-1)**2  # u = 4 - (x - 1)^2 at the top boundary

# Convert K to CSR format for efficient solving
K = csr_matrix(K)

# Solve the linear system
u = spsolve(K, F)

# Reshape the solution for plotting
u_grid = u.reshape((ny+1, nx+1))

# Plot the solution using a contour plot
plt.figure(figsize=(5, 5))
plt.contourf(X, Y, u_grid, 30, cmap='viridis')
plt.colorbar(label="Solution u")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Artificial Diffusion Convection-Diffusion Solution")

# 3D Trisurf Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X.ravel(), Y.ravel(), u, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("3D Artificial Diffusion Convection-Diffusion Solution")
plt.show()
