import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import scienceplots

plt.style.use('science')


# Parameters
nx, ny = 32, 32       # Number of elements in x and y directions
Lx, Ly = 2.0, 2.0     # Domain size
dx, dy = Lx / nx, Ly / ny  # Element sizes

# Diffusion coefficients and convection velocity
epsilon = 1/200         # Physical diffusion coefficient
w = np.array([-np.sin(np.pi/6), np.cos(np.pi/6)])  # Convection velocity [w_x, w_y]

delta = dx/2   # Artificial diffusion coefficient

# Effective diffusion coefficient in the equation: (delta-epsilon)
effective_diff = delta + epsilon

# Generate mesh
x = np.linspace(0, Lx, nx+1)
y = np.linspace(0, Ly, ny+1)
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

# Apply Dirichlet boundary conditions with a discontinuity along one face.
for i in range(num_nodes):
    xi, yi = X.ravel()[i], Y.ravel()[i]
    # Check for nodes on the right boundary (x == Lx)
    if yi == 0:
        # Apply discontinuous condition on the right boundary:
        if xi < 1.0:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 0  # u = 1 for y < 0.5
        else:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 1  # u = 0 for y >= 0.5
    # Apply other boundary conditions as needed (e.g., left, top, bottom)
    # Example: for left boundary (x == 0), you could set u = 1
    elif xi == Lx:
        K[i, :] = 0
        K[i, i] = 1
        F[i] = 1
    elif xi == 0 or yi == Ly:
        K[i, :] = 0
        K[i, i] = 1
        F[i] = 0


# Convert K to CSR format for efficient solving
K = csr_matrix(K)

# Solve the linear system
u = spsolve(K, F)

# Reshape the solution for plotting
u_grid = u.reshape((ny+1, nx+1))

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 4))  
# Create contour plot
contour = ax.contour(X - 1, Y - 1, u_grid, levels=np.linspace(0.0333, 0.9666, 29))
# Add colorbar and link it to the contour plot
cbar = plt.colorbar(contour)
cbar.set_label("u(x)")
# Manually set tick locations
ax.set_xticks(np.linspace(-1, 1, 5))  # Adjusted for X - 1 shift
ax.set_yticks(np.linspace(-1, 1, 5))
ax.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
# Labels
ax.set_xlabel("x", fontsize = 14)
ax.set_ylabel("y", fontsize = 14)
plt.rcParams.update({'font.size': 14})  # Apply globally to all text elements

# 3D Trisurf Plot
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X.ravel(), Y.ravel(), u, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("3D Artificial Diffusion Convection-Diffusion Solution")
ax.view_init(elev=22, azim=-108, roll=0)  # Elevation of 30 degrees, Azimuth of 45 degrees
plt.show()


