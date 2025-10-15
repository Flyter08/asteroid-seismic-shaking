import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree


'''
Use this script as standalone for now.

Let say I have a mesh of an arbitrary shape. I want to find its centre of gravity and go from here.
 - This is not compatible with for example 67p where, two gravitational centres can be described below a global one.

'''

cwd = os.getcwd()
data_folder = "3D models\\v2\\data_folder"
voxel_name = "voxel_dimorphos_decimated_2.5k.stl_n781_001.npz"
voxel_path = os.path.join(cwd, data_folder, voxel_name)
data = np.load(voxel_path)
vertices = data['arr_0']

np.set_printoptions(precision=2) # Print 2 decimals points for numpy arrays
print(f"[INFO - gravity] vertices.shape: {vertices.shape}")
centroid = vertices.mean(axis=0)
print(f"[INFO - gravity] centroid {centroid}")

# Distances from centroid to all vertices
distances = np.linalg.norm(vertices - centroid, axis=1)

print(f"[INFO - gravity] distances: {distances}")
print(f"[INFO - gravity] min distance: {min(distances)}")

# Index of closest vertex
closest_index = np.argmin(distances)

print("[INFO - gravity] Index of vertex closest to centroid:", closest_index)
print("[INFO - gravity] Coordinates:", vertices[closest_index])

rho = 600 # [kg/m³]
diffs = np.diff(vertices, axis=0)#.mean(axis=0) # shape (M-1, 3)
diffs = diffs[diffs >0].mean(axis=0)
diffs = round(diffs * 2) /2
print(f"DIffs {diffs}")
distances_voxel = np.int8(np.array(distances/diffs) +1)
# print(f"[INFO - gravity] # of voxels to centre: {distances_voxel}")
voxel_volume = diffs**3
voxel_mass = np.int32(voxel_volume * rho)
print(f"[INFO - gravity] Voxel volume is {voxel_volume} m³, and mass {voxel_mass/1e3} tonnes.")
cum_mass = np.int32(distances_voxel * voxel_mass)
# print(f"[INFO - gravity] Cum mass to centre {cum_mass/1e3} tonnes")
G = 6.67430e-11 # [m³/(kg s²)] Gravitational constant

gravity_at_vertices = np.zeros_like(vertices)
gravity_at_vertices = (G * cum_mass) / (distances**2)
# print(gravity_at_vertices)
# print(f"[INFO - gravity] Cum grav to centre {gravity_at_vertices} [m/s²]")

total_mass = int(int(voxel_mass) * vertices.shape[0])
total_volume = int(int(voxel_volume) * vertices.shape[0])
print(f"[INFO - gravity] Total mass: {total_mass:.9e} [kg], volume {total_volume:.6e} [m³]")

[Lx, Ly, Lz] = np.max(vertices, axis=0)
print(f"{Lx}, {Ly}, {Lz}")
Nx = int(np.ceil(max([Lx, Ly, Lz])/diffs))
# Nx = 15
Ny, Nz = Nx, Nx
print(f"{Nx}")

# Distribute nodes' position from 0 until the max length
# x = np.linspace(0, Lx, Nx) # [m]
# y = np.linspace(0, Ly, Ny) # [m]
# z = np.linspace(0, Lz, Nz) # [m]
x = np.arange(0, Lx+diffs, diffs)
y = np.arange(0, Ly+diffs, diffs)
z = np.arange(0, Lz+diffs, diffs)

# Precalculate distance between nodes. This is fine for fixed mesh.
dx = np.diff(x)[0] # [m]
dy = np.diff(y)[0] # [m]
dz = np.diff(z)[0] # [m]

# Make a meshgrid indexed by 'ij,k'
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
tree = KDTree(vertices)

distancesTree, indices = tree.query(grid_points)

distancesTree = distancesTree.reshape(X.shape)
threshold = 0.6 * np.array([dx, dy, dz]).max(axis=0)
mask_exists = distancesTree <= threshold
# print(f"Mask exists.shape {mask_exists.shape}")
indices_grid = indices.reshape(X.shape)

gravity_grid = gravity_at_vertices[indices_grid]
# print(f"gravity grid {gravity_grid.shape}")
# gravity_grid = gravity_grid[mask_exists==1]
# print(f"gravity grid {gravity_grid.shape}")


# rho[~mask_exists] = 0


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
scatter = ax.scatter(X[mask_exists], Y[mask_exists], Z[mask_exists], color="orange")
# fig.set(x_label="x", ylabel="y", zlabel="z")

# fig2, ax2 = plt.subplots(1,1, figsize=(6,6))
# grav_plot = ax2.imshow(gravity_grid[:,:,6], cmap="terrain", interpolation=None)
# colorbar = fig2.colorbar(grav_plot, ax=ax2)

# fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
# grplot = ax3.plot(Y[int(X.shape[0]/2),:, int(Z.shape[2]/2)], gravity_grid[int(X.shape[0]/2),:, int(Z.shape[2]/2)])


# fig4, ax4 = plt.subplots(1,2, figsize=(12,6))
cum_mass_grid = cum_mass[indices_grid]
distance_grid = distances[indices_grid]
# cum_plot = ax4[0].imshow(cum_mass_grid[:,5,:], cmap="terrain")
# ax4[0].set_title("Cum mass")
# colorbar = fig4.colorbar(cum_plot, ax=ax4[0])
# distance_plot = ax4[1].imshow(distance_grid[:,5,:], cmap="terrain")
# ax4[1].set_title("Distance to centre")
# colorbar = fig4.colorbar(distance_plot, ax=ax4[1])

# Quick layer model;
r_max = 177/2 # [m]
rho = 600 # kg/m³
mass = [0]
gravity = [0]
dr = 2 # m

radii = np.arange(0,r_max,dr)
for i in range(1,len(radii)):
    shell_volume = 4 * np.pi * radii[i]**2 * dr
    shell_mass = shell_volume * rho
    mass.append(mass[-1] + shell_mass)
    
    g_val = (G * mass[i])/radii[i]**2
    gravity.append(g_val)

fig5, [ax5, ax7] = plt.subplots(1,2, figsize=(12,6))
fig5.suptitle("Gravity and mass models")
ax5.set_title("Sphere layered-model")
grvplot = ax5.plot(gravity, radii)
ax5.set_xlabel("Gravitational acceleration [m/s²]")
ax5.set_ylabel("Radius from centre [m]")
ax52 = ax5.twiny()
ax52.set_xlabel("Cumultative mass from centre [kg]")
ax5.grid()

grvplot = ax52.plot(mass, radii, color="tab:orange")

print(int(Nx/2)) # 6 ok lets say
print(f"Og centroid {centroid}")
mass = [0.0]
gravity = [0.0]
previous_neighbors = []
for i in range(0,int(Nx/2)):
    distancesTree, neighbor_indices = tree.query(centroid, k=7)
    print(f"neighbors {neighbor_indices}")
    
    # previous_neighbors = np.array(previous_neighbors)
    # neighbor_indices = neighbor_indices.flatten()
    # print(type(neighbor_indices), np.shape(neighbor_indices))
    # print(type(previous_neighbors), np.shape(previous_neighbors))
    

    
    if i == 0:
        previous_neighbors.append(int(neighbor_indices[0]))
        print(f"previous neighbors {previous_neighbors}")
        neighbor_indices = neighbor_indices[1:]
        previous_neighbors_arr = np.array(previous_neighbors)
        
    else:
        previous_neighbors.extend(neighbor_indices[:,0])
        print(f"previous neighbors {previous_neighbors}")
        previous_neighbors_arr = np.array(previous_neighbors)
        # neighbor_indices = np.unique([neighbor_indices, previous_neighbors])#[(centroid.shape[-1]):] # removes self (centroid)
        neighbor_indices_flat = neighbor_indices.ravel()
        combined = np.concatenate((neighbor_indices_flat, previous_neighbors_arr))
        neighbor_indices = np.unique(combined)

    
    print(f"Decimated neighbors {neighbor_indices}")
    mass.append(mass[-1] + neighbor_indices.shape[0] * int(voxel_volume) * rho)
    
    gravity.append(((G*mass[-1])/((i+1)*dx)**2))
    gravity_at_vertices[neighbor_indices] = gravity[-1]
    print(f"iteration {i}, mass[-1]: {mass[-1]} and new grav {gravity[-1]}")
    centroid = vertices[neighbor_indices,:]
    # print(f"New centroid {centroid}")
# print(mass)
print(f"Mass max = {mass[-1]:.9e}")
# print(len(mass))
# print(mass)
# print(len(gravity))
# print(gravity)
radii = np.arange(0, Lx/2, Nx)
# fig7, ax7 = plt.subplots(1,1, figsize=(6,6))
ax7.set_title("Voxeled model")
ax7.grid()
ax7.set_xlabel("Gravitational acceleration [m/s²]")
ax7.set_ylabel("Radius from centre [m]")
grvplot = ax7.plot(gravity, radii)
ax72 = ax7.twiny()
ax72.set_xlabel("Cumultative mass from centre [kg]")
grvplot = ax72.plot(mass, radii, color="tab:orange")



fig8 = plt.figure(figsize=(6,6))
ax8 = fig8.add_subplot(111, projection='3d')
print(f"Gravity vertices{gravity_at_vertices.shape}")
colors = np.array(gravity_at_vertices.copy(), dtype = float)
print(f"Colors mins: {colors.min()}, max: {colors.max()}, rand colors {colors[[231,565,0,324,765]]}")
gravity_grid = gravity_at_vertices[indices_grid]
print(f"Gravity grid{gravity_grid.shape}")
# colors = gravity_grid[mask_exists]
# colors_norm = (colors - colors.min()) / np.ptp(colors)
colors_norm = colors / colors.max()
print(colors_norm)
colors_norm_grid = colors_norm[indices_grid]
# print(f"COlors norm{colors_norm}")
# scatter = ax8.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
scatter = ax8.scatter(X[mask_exists], Y[mask_exists], Z[mask_exists], c=colors_norm_grid[mask_exists], cmap="viridis")
fig8.colorbar(scatter, ax=ax8)
# fig6 = plt.figure(figsize=(6,6))
# ax6 = fig6.add_subplot(121, projection='3d')
# ax61 = fig6.add_subplot(122, projection='3d')
# colors = cum_mass_grid[mask_exists]
# colors_norm = (colors - colors.min()) / (colors.max() - colors.min())
# cum_scatter = ax6.scatter(X[mask_exists], Y[mask_exists], Z[mask_exists], c=colors_norm, cmap="viridis", s=50)

# colors = gravity_grid[mask_exists]
# colors_norm = (colors - colors.min()) / (colors.max() - colors.min())
# cum_scatter = ax61.scatter(X[mask_exists], Y[mask_exists], Z[mask_exists], c=colors_norm, cmap="viridis", s=50)

plt.tight_layout()

plt.show()