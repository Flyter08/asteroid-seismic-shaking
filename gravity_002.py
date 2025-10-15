import numpy as np
import os
from polyhedral_gravity import Polyhedron, GravityEvaluable, evaluate, PolyhedronIntegrity, NormalOrientation, MetricUnit
import mesh_plotting
from matplotlib import pyplot as plt
import trimesh

cwd = os.getcwd()
print(cwd)
file_path = os.path.join(cwd, "3D models\\v2\\models\\dimorphos_decimated_25k.stl")

computation_point = np.array([0, 0, 0])
if os.path.exists(file_path):
    print(f"ok")
density = 600 # [kg/m³]

# mesh = trimesh.load(file_path)
# scale_factor = 1000
# mesh.apply_scale(scale_factor)
file_path_export = os.path.join(cwd, "3D models\\v2\\models\\dimorphos_decimated_25k_scaled.stl")
# mesh.export(file_path_export)
# Correct instantiation
dimorphos_polyhedron = Polyhedron(
    # polyhedral_source=["C:\\Users\\Flyte\\OneDrive - Delft University of Technology\\Internship\\code\\3D models\\v2\\models\\dimorphos_decimated_25k.stl"],
    polyhedral_source=[file_path_export],
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
closest_index = np.abs(Z - dimorphos_centre[2]).argmin()
print(f"closest index {closest_index} @ {Z[closest_index]}")
# print(f"Size X {X.size}, size Y {Y.size}")

computation_points = np.array(np.meshgrid(X, Y, closest_index)).T.reshape(-1, 3)
gravity_results = evaluable_dimorphos(computation_points)

potentials = -1 * np.array([i[0] for i in gravity_results])
# print(f"POTENTIALS {potentials.size}")
# print(f"Gravity results size {len(gravity_results)}")
potentials = potentials.reshape((len(X), len(Y)))

X = computation_points[:, 0].reshape(len(X), -1)
Y = computation_points[:, 1].reshape(len(Y), -1)

# mesh_plotting.plot_grid_2d(X, Y, potentials, "Potential of Dimorphos", plot_rectangle=False)


accelerations = np.array([i[1][:] for i in gravity_results])
# print(accelerations.size)
accelerations_norm = np.linalg.norm(accelerations[:,:2], axis=1)
# print(accelerations_norm)
# print(len(accelerations_norm))
acc_xy = np.delete(accelerations, 2, 1)

mesh_plotting.plot_quiver(X, Y, acc_xy, "Acceleration in $x$ and $y$ direction for $z=0$", vertices=np.array(dimorphos_polyhedron.vertices), coordinate=2)
# print(acc_xy)
accelerations_norm = accelerations_norm.reshape((len(X), len(Y)))

axx = mesh_plotting.plot_grid_2d(X, Y, accelerations_norm, "Acceleration in $x$ and $y$.", vertices=np.array(dimorphos_polyhedron.vertices), coordinate=2)

# mesh_plotting.plot_polygon(axx, dimorphos_polyhedron.vertices, 1)

# mesh_plotting.plot_triangulation(np.array(dimorphos_polyhedron.vertices), list(dimorphos_polyhedron.faces), "The triangulation of the $[-1,1]^3$ cube")

plt.show()