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
    v = np.array([0 , 1])

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
    x = np.linspace(-1, 1, nx+1)
    y = np.linspace(-1, 1, ny+1)
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
        # Apply Dirichlet condition u = 0 on the bottom boundary (y = 0)
        if yi == 0:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = xi  # u = 0 at the bottom boundary
        
        # Apply Dirichlet condition u = 4 - (x - 1)^2 on the top boundary (y = L_y)
        elif yi == 1:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 0  # u = 4 - (x - 1)^2 at the top boundary

        elif xi == -1:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = -1  # u = 0

        elif xi == 1:
            if yi != Ly:
                K[i, :] = 0
                K[i, i] = 1
                F[i] = 1


    # Convert to sparse format
    K = csr_matrix(K)

    # Solve the linear system
    u = spsolve(K, F)

    # Reshape for plotting
    u_grid = u.reshape((ny+1, nx+1))
    
    return X, Y, u_grid, x, y, u


def plot_contour():
    def plotter_c(fill, ax, method, title):
        X, Y, u_grid, x, y, u = Solve(method)
        
        num_contours = 11
        contours = np.linspace(0, 1, num_contours)
        #contours = contours[1:num_contours-1]
        if fill == True:
            contour = ax.contourf(X - 1, Y - 1, u_grid)
        elif fill == False:
            contour = ax.contour(X - 1, Y - 1, u_grid)
        
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
    fill = True
    plotter_c(fill, axes[0, 0], method="Galerkin", title="Galerkin")
    plotter_c(fill, axes[0, 1], method="Artificial", title="Artificial")
    plotter_c(fill, axes[1, 0], method="SUPG", title="SUPG")

    # Leave the fourth subplot empty
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Plot 3D function
def plot_3d():
    def plotter3d(ax, method, title):
        # Get data from Solve; it must return at least X, Y, and u
        X, Y, u_grid, x, y, u = Solve(method)
        # Plot the 3D surface using trisurf
        ax.plot_trisurf(X.ravel(), Y.ravel(), u, cmap='jet', edgecolor='k')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_title(title)
        ax.view_init(elev=22, azim=-108, roll=0)  # adjust view as needed

    # Create a figure and add 3D subplots manually
    fig = plt.figure(figsize=(8, 6))

    # Create three 3D subplots in a 2x2 grid and leave the fourth blank
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)  # this will remain blank
    ax4.axis('off')

    # Plot the three different methods
    plotter3d(ax1, method="Galerkin", title="Galerkin")
    plotter3d(ax2, method="Artificial", title="Artificial Diffusion")
    plotter3d(ax3, method="SUPG", title="SUPG")

    plt.tight_layout()
    plt.show()

plot_3d()