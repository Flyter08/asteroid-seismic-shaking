import numpy as np
import matplotlib.pyplot as plt

G = 6.67430e-11  # gravitational constant (m^3/kg/s^2)
M = 1e12         # asteroid mass in kg (example)
R = 500          # asteroid radius in meters

size = 2000      # extent of grid in meters
n = 40           # grid resolution

x = np.linspace(-size/2, size/2, n)
y = np.linspace(-size/2, size/2, n)
X, Y = np.meshgrid(x, y)

dist = np.sqrt(X**2 + Y**2)
dist[dist < R] = R  # avoid singularity inside asteroid, assume constant surface gravity

g_mag = G * M / dist**2  # gravity magnitude

g_x = -g_mag * X / dist  # gravity vector x-component (pointing inward)
g_y = -g_mag * Y / dist  # gravity vector y-component

plt.figure(figsize=(8,8))
plt.quiver(X, Y, g_x, g_y, g_mag, cmap='inferno', scale=3e6)
plt.colorbar(label='Gravity magnitude (m/s²)')
plt.title('Gravity gradient vector field around spherical asteroid (2D slice)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.axis('equal')
plt.show()
