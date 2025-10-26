import numpy as np
import matplotlib.pyplot as plt

# Set the diffusion parameter epsilon
epsilon = 0.05

# Create a grid over the unit square
N = 20  # Number of grid points in each direction
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Compute the exact solution u(x,y)
# Avoid numerical issues: when epsilon is very small, the denominator can be close to 0.
denom = 1 - np.exp(-2/epsilon)
u = ((X)**3+1)*(1 - np.exp((Y - 1)/epsilon)) / denom

print(np.max(u))
print(np.min(u))

# Create a contour plot
plt.figure(figsize=(5, 5))
contour = plt.contourf(X, Y, u, levels=8, cmap='viridis')
plt.colorbar(contour)
plt.xlabel('x')
plt.ylabel('y')

# Create a figure and 3D axis
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, u, cmap='viridis', edgecolor='none', rstride=1, cstride =1)

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
ax.set_title('3D Surface Plot of Exact Solution')
ax.view_init(elev=25, azim=145)

plt.show()
