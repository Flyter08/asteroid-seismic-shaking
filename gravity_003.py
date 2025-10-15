import numpy as np
import os
from polyhedral_gravity import Polyhedron, GravityEvaluable, evaluate, PolyhedronIntegrity, NormalOrientation, MetricUnit
import mesh_plotting
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
import trimesh
from scipy.spatial import KDTree

cwd = os.getcwd() # Get the current working directory on user's machine
print(f"Currently working from: {cwd}")
stl_file = "dimorphos_decimated_25k.stl"
models_path = os.path.join(cwd, "models")
data_path = os.path.join(cwd, "data_folder")
stl_path = os.path.join(models_path, stl_file)
voxels_file = "voxel_dimorphos_decimated_2.5k.stl_n781_001.npz"
voxels_path = os.path.join(data_path, voxels_file)
save_file = False

computation_point = np.array([0, 0, 0])

if os.path.exists(stl_path):
    print(f"\033[32m[INFO - gravity]\033[0m .stl source file <{stl_file}>.")
    scaled_stl_file = os.path.splitext(stl_file)[0] + "_scaled.stl"
    scaled_stl_path = os.path.join(models_path, scaled_stl_file)
    if os.path.exists(scaled_stl_path):
        print(f"\033[32m[INFO - gravity]\033[0m Found scaled.stl file <{scaled_stl_file}>.")
    else:
        scale_factor = 1000
        print(f"\033[32m[INFO - gravity]\033[0m Could not find scaled.stl file <{scaled_stl_file}>.\nScaling source file by factor {scale_factor}...")
        mesh = trimesh.load(stl_path)
        mesh.apply_scale(scale_factor)
        mesh.export(scaled_stl_path)
else:
    print(f"\033[91m[ERROR - gravity]\033[0m .stl source file <{stl_file}> not found in <{models_path}>")


density = 600 # [kg/m³]

dimorphos_polyhedron = Polyhedron(
    polyhedral_source=[scaled_stl_path],
    density=density,
    normal_orientation=NormalOrientation.INWARDS,
    integrity_check=PolyhedronIntegrity.HEAL,
    metric_unit=MetricUnit.METER
)
potential, acceleration, tensor = evaluate(
  polyhedron=dimorphos_polyhedron,
  computation_points=computation_point,
  parallel=True,
)
print("Potential: {:.4e}".format(potential))
print("Acceleration [Vx, Vy, Vz]: {}".format(acceleration))
print("Second derivative tensor [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz]: {}".format(tensor))

evaluable_dimorphos = GravityEvaluable(dimorphos_polyhedron)
print(evaluable_dimorphos)
# We use for this plot a more coarse grid to improve the visualization
dim = np.array([177, 174, 116]) # [m] Actual size of asteroid in metres
dimorphos_centre = dim/2
print(f"Dimorphos centre located at {dimorphos_centre} [m]")
X = np.arange(-50, 230, 5)
Y = np.arange(-50, 230, 5)
Z = np.arange(-50, 230, 5)
mesh_list = [X, Y, Z]
plane_names = ['YZ', 'XZ', 'XY']
labels = ["x", "y", "z"]

for axis in range(3):
    idx = [i for i in range(3) if i != axis]
    print(f"idx {idx}, axis {axis}")
    x_plane = mesh_list[idx[0]]
    y_plane = mesh_list[idx[1]]

    closest_index = np.abs(mesh_list[axis] - dimorphos_centre[axis]).argmin()
    print(f"closest index {closest_index} @ {mesh_list[axis][closest_index]}")
    # print(f"Size X {X.size}, size Y {Y.size}")
    if axis == 0:
        computation_points = np.array(np.meshgrid(closest_index, x_plane, y_plane)).T.reshape(-1, 3)
    elif axis == 1:
        computation_points = np.array(np.meshgrid(x_plane, closest_index, y_plane)).T.reshape(-1, 3)
    elif axis == 2:
        computation_points = np.array(np.meshgrid(x_plane, y_plane, closest_index)).T.reshape(-1, 3)
    gravity_results = evaluable_dimorphos(computation_points)

    accelerations = np.array([i[1][:] for i in gravity_results])

    accelerations_norm = np.linalg.norm(accelerations[:,idx], axis=1)
    
    # acc_xy = np.delete(accelerations, 2, 1)

    accelerations_norm = accelerations_norm.reshape((len(x_plane), len(y_plane)))

    mesh_plotting.plot_grid_2d(
        X=x_plane, 
        Y=y_plane, 
        z=accelerations_norm, 
        title=f"Acceleration in plane: {plane_names[axis]}; axis: {axis}.",  
        labels=(str(labels[idx[0]]), str(labels[idx[1]])),
        vertices=np.array(dimorphos_polyhedron.vertices),
        coordinate=axis,
        plot_rectangle=True,
        dim=dim[idx]
        )



# plt.show()

do_3d = True
if do_3d:
    X = np.arange(0, dim[0], 10)
    Y = np.arange(0, dim[0], 10)
    Z = np.arange(0, dim[0], 10)
    # Precalculate distance between nodes. This is fine for fixed mesh.
    dx = np.diff(X)[0] # [m]
    dy = np.diff(Y)[0] # [m]
    dz = np.diff(Z)[0] # [m]

    computation_points = np.array(np.meshgrid(X, Y, Z, indexing='xy')).T.reshape(-1, 3) # compute over all points in mesh

    gravity_results = evaluable_dimorphos(computation_points)
    accelerations = np.array([i[1][:] for i in gravity_results])
    accelerations_norm = np.linalg.norm(accelerations[:], axis=1) # all accelerations since 3D


    X, Y, Z = np.meshgrid(X, Y, Z, indexing='xy')
    accelerations_norm = accelerations_norm.reshape(X.shape).T # reshape into meshgrid format


    # Make mask for voxel centres
    data = np.load(voxels_path) # Load the numpyzip file
    points = data['arr_0']
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    tree = KDTree(points)
    distances, indices = tree.query(grid_points)

    distances = distances.reshape(X.shape)
    threshold = 3 * np.array([dx, dy, dz]).max(axis=0)
    mask_exists = distances <= threshold
    fig, ax = plt.subplots(1,1,figsize=(5,5), subplot_kw={"projection": "3d"})
    mask = (
        (X>=0) & (X<=dim[0]) &
        (Y>=0) & (Y<=dim[1]) &
        (Z>=0) & (Z<=dim[2])
        )

    ax.scatter(X[mask_exists], Y[mask_exists], Z[mask_exists], c=accelerations_norm[mask_exists], cmap="terrain", s=20, alpha=1)
    ax.set(xlabel=f"x ({dim[0]}) [m]", ylabel=f"y ({dim[1]}) [m]", zlabel=f"z ({dim[2]}) [m]")


# Save gravitational acceleration magnitude into numpyz file.
gravity_file = "gravity_magnitude_001.npz"
gravity_path = os.path.join(data_path, gravity_file)

if not os.path.exists(gravity_path) and save_file:
    print(f"\033[32m[INFO - gravity]\033[0m Saving acceleration_norm <{accelerations_norm.shape}> <shape> at <{gravity_path}>...")   
    np.savez(gravity_path, gravity_magnitude=accelerations_norm)
    print(f"\033[32m[INFO - gravity]\033[0m Successfully saved acceleration_norm <{accelerations_norm.shape}> <shape> at <{gravity_path}>...")  
else: 
    print(f"\033[32m[INFO - gravity]\033[0m acceleration_norm <{accelerations_norm.shape}> <shape> already exists or no save requested.")   

plt.show()




