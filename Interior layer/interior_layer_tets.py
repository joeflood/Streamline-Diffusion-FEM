import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import scienceplots

plt.style.use('science')

def Solve(method):
    # Parameters
    nx, ny = 24, 24   # Number of elements in x and y directions
    Lx, Ly = 2.0, 2.0 # Domain size
    dx, dy = Lx / nx, Ly / ny  # Element sizes
    gamma = 1/200  # Diffusion coefficient
    v = np.array([-np.sin(np.pi/6), np.cos(np.pi/6)])

    if method == "SUPG":
        print("Method: SUPG")
        tau = dx/2  # SUPG stabilization parameter
    elif method == "Galerkin":
        print("Method: Galerkin")
        tau = 0
    elif method == "Artificial":
        print("Method: Artificial")
        gamma = gamma + dx/2
        tau = 0

    # Generate mesh
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    X, Y = np.meshgrid(x, y)

    # Number of nodes
    num_nodes = (nx+1) * (ny+1)

    # Define the source function
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


    # Convert to sparse format
    K = csr_matrix(K)

    # Solve the linear system
    u = spsolve(K, F)

    # Reshape for plotting
    u_grid = u.reshape((ny+1, nx+1))
    
    return X, Y, u_grid

def plot_solution(ax, method, title):
    X, Y, u_grid = Solve(method)
    
    num_contours = 31
    contours = np.linspace(0, 1, num_contours)
    contours = contours[1:num_contours-1]

    contour = ax.contour(X - 1, Y - 1, u_grid, levels=contours)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("u(x)")
    
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))
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