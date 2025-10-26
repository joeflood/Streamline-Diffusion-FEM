import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Parameters
nx, ny = 10, 10  # Number of elements in x and y directions
Lx, Ly = 1.0, 1.0  # Domain size
dx, dy = Lx / nx, Ly / ny  # Element sizes
gamma = 1/200  # Diffusion coefficient
v = np.array([1, 1])  # Convection velocity [vx, vy]
tau = dx/2  # SUPG stabilization parameter

# Mesh Generation with Boundary Layer
"""
boundary_layer_refinement = 5  # Number of elements to refine on one side
nx_left = nx - boundary_layer_refinement  # Elements excluding the boundary layer part
nx_right = boundary_layer_refinement
#}

# Refine mesh near x=0 for boundary layer
x_left = np.linspace(0, 0.2, nx_left + 1)
x_right = np.linspace(0.2, 1.0, nx_right + 1)
x = np.concatenate([x_left[1:], x_right])"
"""

# Create mesh grid with boundary layer refinement
x = np.linspace(0, Lx, nx + 1)
y = np.linspace(0, Ly, ny + 1)

#X, Y = np.meshgrid(x, y)

# Number of nodes
num_nodes = (nx + 1) * (ny + 1)

# Define the source function
f = np.zeros(num_nodes)

# Global stiffness matrix and load vector
K = lil_matrix((num_nodes, num_nodes))
F = np.zeros(num_nodes)

# Map from 2D (i,j) to global node number
def node_num(i, j):
    return j * (nx + 1) + i

# Standard bilinear shape functions and gradients for quadrilaterals
grad_N = np.array([[-1/dx, -1/dy], [1/dx, -1/dy], [1/dx, 1/dy], [-1/dx, 1/dy]])

# Assembly loop
for i in range(nx):
    for j in range(ny):
        # Get global node numbers of the element
        nodes = [node_num(i, j), node_num(i+1, j), node_num(i+1, j+1), node_num(i, j+1)]
        xc, yc = x[i] + dx/2, y[j] + dy/2  # Element center

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

def apply_boundary_conditions():
    for i in range(num_nodes):
        xi, yi = X.ravel()[i], Y.ravel()[i]  # Get x, y coordinates of the node
        # Left boundary (x = 0)
        if np.isclose(xi, 0):  # Use np.isclose to check boundary near x = 0
            K[i, :] = 0    # Zero out the row
            K[i, i] = 1    # Set diagonal to 1
            F[i] = 1       # Set value to 1

        # Other boundaries (right boundary x = Lx, bottom y = 0, top y = Ly)
        elif np.isclose(xi, Lx) or np.isclose(yi, 0) or np.isclose(yi, Ly):
            K[i, :] = 0    # Zero out the row
            K[i, i] = 1    # Set diagonal to 1
            F[i] = 0       # Set value to 0

apply_boundary_conditions()

# Convert to sparse format
K = csr_matrix(K)

# Solve the linear system
u = spsolve(K, F)

# Reshape for plotting
u_grid = u.reshape((ny+1, nx+1))

# Plot the solution using imshow (structured quadrilateral grid)
plt.figure(figsize=(8, 6))
plt.imshow(u_grid, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label="Solution u")
plt.xlabel("x")
plt.ylabel("y")
plt.title("SUPG-Stabilized Convection-Diffusion Solution")
plt.show()

# Plot the 3D surface plot using quadrilateral elements
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
X_flat = X.ravel()
Y_flat = Y.ravel()
u_flat = u.ravel()

# Use plot_surface instead of plot_trisurf
X_grid, Y_grid = np.meshgrid(x, y)
ax.plot_surface(X_grid, Y_grid, u_grid, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("3D Surface Plot with Quadrilateral Elements")
plt.show()


# Plot the 3D surface plot using quadrilateral elements
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
X_flat = X.ravel()
Y_flat = Y.ravel()
u_flat = u.ravel()

# Use plot_surface instead of plot_trisurf
X_grid, Y_grid = np.meshgrid(x, y)
ax.plot_surface(X_grid, Y_grid, u_grid, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("3D Surface Plot with Quadrilateral Elements")
plt.show()