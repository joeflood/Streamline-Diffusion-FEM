import numpy as np
import matplotlib.pyplot as plt

n = 5

# Function to create a hat function (piecewise linear)
def hat_function(x, center, width=1/n):
    return np.maximum(0, 1 - np.abs((x - center) / width))

# Create the x values (interval 0 to 1)
x = np.linspace(0, 1, 500)

# Plot multiple hat functions with different centers and widths
plt.figure(figsize=(8, 6))

centres = np.linspace(0,1,n+1)
print(centres)
for centre in centres:
    plt.plot(x, hat_function(x, centre), label=f'Center = {centre:.2f}')

plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.xticks(centres)
plt.yticks([0, 1])
plt.grid(False)  # Enable grid only for the y-axis (horizontal lines)
plt.show()
