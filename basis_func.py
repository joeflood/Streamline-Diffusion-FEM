import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y)

# Define bilinear basis functions
phi_00 = (1 - X) * (1 - Y)  # Node at (0,0)
phi_10 = X * (1 - Y)        # Node at (1,0)
phi_01 = (1 - X) * Y        # Node at (0,1)
phi_11 = X * Y              # Node at (1,1)

# Plotting
fig = plt.figure(figsize=(12,4))

basis_functions = [phi_00, phi_10, phi_01, phi_11]

for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    ax.plot_surface(X, Y, basis_functions[i], cmap='viridis', edgecolor='none')

    
    # Set ticks only at 0 and 1
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

        # Replace '1' with 'h' on x, y, and z axes
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])

    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    ax.view_init(elev=30, azim=-135)  # 3D view

plt.tight_layout()
plt.show()
