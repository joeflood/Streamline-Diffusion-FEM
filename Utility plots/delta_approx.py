import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

# Create the x values (interval 0 to 1)
x = np.linspace(0, 5, 100)

def coth(x):
    return np.cosh(x) / np.sinh(x)

exact = (coth(x) - 1/x)

xa     = np.linspace(0, 5, 100)
approx = np.zeros_like(xa)

for idx, xval in enumerate(xa):
    if xval <= 1:
        approx[idx] = 0.0
    else:
        approx[idx] = 1.0 - 1.0/xval


# Plot multiple hat functions with different centers and widths

# Create figure and axis
fig, ax = plt.subplots(figsize = (6,3))


# Plot the function
ax.plot(x, exact, xa, approx, '--')

# Set labels
ax.set_xlabel('$Pe$', fontsize = 14)
ax.set_ylabel(r'$\hat{\epsilon}_{\text{total}}$', fontsize = 14)

# Adjust limits and margins
ax.set_ylim(-1, 1)
ax.margins(x=0, y=0)

# Modify spines to align axes properly
ax.spines['bottom'].set_position(('data', 0))  # Move x-axis to y=0
ax.spines['bottom'].set_linewidth(plt.rcParams['axes.linewidth'])  # Match linewidth
ax.spines['left'].set_linewidth(plt.rcParams['axes.linewidth'])

# Hide the top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)


plt.show()