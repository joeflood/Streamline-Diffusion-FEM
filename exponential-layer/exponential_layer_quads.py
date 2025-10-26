import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import dblquad
import matplotlib.ticker as ticker
import scienceplots

plt.style.use('science')

def solve(method):
    # Parameters
    nx, ny = 24, 24           # Number of elements in x and y directions
    xmin, xmax, ymin, ymax = -1, 1, -1, 1 #xmin, xmax, ymin, ymax
    Lx, Ly = xmax - xmin, ymax - ymin # Domain size
    dx, dy = Lx / nx, Ly / ny  # Element sizes
    epsilon = 1/2400            # Diffusion coefficient
    v = np.array([0, 1])
    Peclet = (1*dx)/(2*epsilon)
    print("Mesh Peclet number = " + str(Peclet))
    print(1 - 1/Peclet)
    print(1/np.tanh(Peclet) - 1/Peclet)

    if method == "SUPG":
        print("Method: SUPG")
        tau = dx/2 * (1-1/Peclet)  # SUPG stabilization parameter
    elif method == "Galerkin":
        print("Method: Galerkin")
        tau = 0
    elif method == "Artificial":
        print("Method: Artificial")
        epsilon = epsilon + dx/2
        tau = 0

    # Create uniform mesh
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
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
                    diffusive_term   = epsilon * np.dot(grad_N[a], grad_N[b]) * dx * dy
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
    for i in range(num_nodes):
        xi, yi = X.ravel()[i], Y.ravel()[i]
        if yi == ymin:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = xi**3 + 1 
        elif yi == ymax:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 0 
        elif xi == xmin:
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 0
        elif xi == xmax:
            if yi != Ly:
                K[i, :] = 0
                K[i, i] = 1
                F[i] = 2

    # Convert stiffness matrix to CSR format and solve the linear system
    K = csr_matrix(K)
    u = spsolve(K, F)
    u_grid = u.reshape((ny + 1, nx + 1))

    # Compute gradient per element (piecewise constant per element)
    grad_u = np.zeros((ny, nx, 2))  # 2 for [du/dx, du/dy]
    for i in range(nx):
        for j in range(ny):
            # Nodes for the current element
            nodes = [node_num(i, j), node_num(i+1, j), node_num(i+1, j+1), node_num(i, j+1)]
            # Compute the gradient in the element as the sum of nodal values times shape function gradients
            grad_u[j, i, :] = (u[nodes[0]] * grad_N[0] +
                               u[nodes[1]] * grad_N[1] +
                               u[nodes[2]] * grad_N[2] +
                               u[nodes[3]] * grad_N[3])
    # grad_u now contains the gradient (du/dx, du/dy) for each element

    return X, Y, u_grid, x, y, grad_u

# --------------------- Plotting ---------------------

def plot(type):
    def plotter_c(fill, ax, method, title):
        X, Y, u_grid, x, y, grad_u = solve(method)
        num_contours = 31
        contours = np.linspace(-1, 1, num_contours)
        contours = contours[1:num_contours-1]

        if fill == True:
            contour = ax.contourf(X, Y, u_grid, levels = contours)
        elif fill == False:
            contour = ax.contour(X, Y, u_grid, levels = contours)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("u(x)")
        ax.set_xticks(np.linspace(-1, 1, 5))
        ax.set_yticks(np.linspace(-1, 1, 5))
        ax.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

    def plotter3d(ax, method, title):
        # Get data from Solve; it must return at least X, Y, and u
        X, Y, u_grid, x, y, grad_u = solve(method)
        # Plot the 3D surface 
        ax.plot_surface(X, Y, u_grid, cmap = 'viridis', edgecolor='k')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        #ax.set_zlabel("u")
        ax.set_zlim(0, 2.5)
        ax.set_title(title)
        ax.view_init(elev=25, azim=160)
        ax.margins(0)
        ax.tick_params(axis='x', pad=0.3)  # Bring x-axis tick labels closer
        ax.tick_params(axis='y', pad=0.3)  # Bring y-axis tick labels closer
        ax.tick_params(axis='z', pad=0.3)  # Bring z-axis tick labels closer

    def plot_exact(ax, method, title):
        X, Y, u_grid, x, y, grad_u = solve(method)
        epsilon = 1/2400
        u = ((X)**3+1)*(1 - np.exp((Y - 1)/epsilon)) / (1 - np.exp(-2/epsilon))
        ax.plot_surface(X, Y, u, cmap = 'viridis', edgecolor='k')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        #ax.set_zlabel("u")
        ax.set_zlim(0, 2.5)
        ax.set_title(title)
        ax.view_init(elev=25, azim=160)
        ax.margins(0)
        ax.tick_params(axis='x', pad=0.3)  # Bring x-axis tick labels closer
        ax.tick_params(axis='y', pad=0.3)  # Bring y-axis tick labels closer
        ax.tick_params(axis='z', pad=0.3)  # Bring z-axis tick labels closer


    if type == "contourf":
        fig, axes = plt.subplots(1, 3, figsize=(8, 6))
        fill = True
        plotter_c(fill, axes[0], method="Galerkin", title="Galerkin")
        plotter_c(fill, axes[1], method="SUPG", title="SUPG")
        plot_exact(axes[2], method="SUPG", title = "Exact")
        plt.tight_layout()
        plt.show()

    elif type == "contour":
        fig, axes = plt.subplots(1, 3, figsize=(8, 6))
        fill = False
        plotter_c(fill, axes[0], method="Galerkin", title="Galerkin")
        plotter_c(fill, axes[1], method="SUPG", title="SUPG")

        plt.tight_layout(pad = 1)
        plt.show()

    elif type == "3d":
            # Create a figure and add 3D subplots manually
        fig = plt.figure(figsize=(9, 3))

        # Create three 3D subplots in a 2x2 grid and leave the fourth blank
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        # Plot the three different methods
        plotter3d(ax1, method="Galerkin", title="Galerkin")
        plotter3d(ax2, method="SUPG", title="SUPG")
        plot_exact(ax3, method="SUPG", title = "Exact")

        plt.tight_layout()
        plt.show()

# Plot 3D function
plot("3d")

def exact_grad(x, y):
    epsilon = 1/2400
    denom = 1 - np.exp(-2/epsilon)
    u_x = 3*x**2*(1-np.exp((y-1)/epsilon))/denom
    u_y = (x**3 + 1)*((1-y)/epsilon)*np.exp((y-1)/epsilon)/denom
    return np.array([u_x,
                     u_y])


def interpolated_grad(xp, yp, grad_u, x, y):
    """
    Given a point (xp, yp), return the FE gradient for the element
    that contains the point.
    """
    # Find the indices: we use np.searchsorted to locate the interval.
    # Note: x and y are 1D arrays of node coordinates.
    i = np.searchsorted(x, xp, side='right') - 1
    j = np.searchsorted(y, yp, side='right') - 1

    # Clamp indices to valid range (if xp == x[-1] or yp == y[-1])
    if i >= len(x) - 1:
        i = len(x) - 2
    if j >= len(y) - 1:
        j = len(y) - 2

    # Return the gradient for element at indices (j, i)
    return grad_u[j, i, :]

def integrand(yp, xp, grad_u, x, y):
    """
    The integrand function for dblquad.
    Note: dblquad expects the function to have signature f(y, x).
    """
    g_exact = exact_grad(xp, yp)
    g_fe = interpolated_grad(xp, yp, grad_u, x, y)
    diff = g_exact - g_fe
    return diff[0]**2 + diff[1]**2

def compute_L2_grad_error_dblquad(grad_u, x, y, epsabs=1e-13, epsrel=1e-13):
    """
    Compute the L2 norm of the gradient difference using adaptive quadrature.
    
    Parameters:
      grad_u : (ny, nx, 2) array of FE gradients per element.
      x      : 1D array of x coordinates (length = nx+1).
      y      : 1D array of y coordinates (length = ny+1).
    
    Returns:
      error : L2 norm of the gradient difference.
    """
    # Domain limits (assuming the full domain is covered)
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    
    # Use dblquad to integrate the squared error over the domain.
    # Note: the lambda function reorders arguments as dblquad calls f(yp, xp).
    result, err = dblquad(
        lambda yp, xp: integrand(yp, xp, grad_u, x, y),
        xmin, xmax,
        lambda xp: ymin, lambda xp: ymax,
        epsabs=epsabs, epsrel=epsrel
    )
    
    # The L2 norm is the square root of the integral.
    return np.sqrt(result)

# Example usage:
X, Y, u_grid, x, y, grad_u = solve("Galerkin")
error = compute_L2_grad_error_dblquad(grad_u, x, y)
print("L2 norm of the gradient error (dblquad):", error)

X, Y, u_grid, x, y, grad_u = solve("SUPG")
error = compute_L2_grad_error_dblquad(grad_u, x, y)
print("L2 norm of the gradient error (dblquad):", error)