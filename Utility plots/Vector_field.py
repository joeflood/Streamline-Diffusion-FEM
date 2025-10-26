import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
U = 2*Y*(1-X**2)  # Example velocity in X-direction
V = -2*X*(1-Y**2)   # Example velocity in Y-direction

plt.quiver(X, Y, U, V, width = 0.004)
plt.xticks([])  # Remove x-axis numbers
plt.yticks([])  # Remove y-axis numbers
plt.show()
