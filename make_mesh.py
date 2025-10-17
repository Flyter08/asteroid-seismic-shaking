import os
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import KDTree, ConvexHull
# import time
def get_physical_params(physical_params):
    # Unpack variables from packed params
    # Assertion for physical parameters: all keys are present; if not, AssertionError with message raised
    expected_keys = ["dt", "lamb", "mu", "rho", "gamma", "default_normal_stress"]

    if all(key in physical_params for key in expected_keys):
        dt = physical_params["dt"]
        lamb = physical_params["lamb"]
        mu = physical_params["mu"]
        rho = physical_params["rho"]
        gamma = physical_params["gamma"]
        default_normal_stress = physical_params["default_normal_stress"]
        return dt, lamb, mu, rho, gamma, default_normal_stress
    else:
        print(f"\033[91m[ERROR - mesh]\033[0m Missing keys: {[key for key in expected_keys if key not in physical_params]}.\n\033[91mAborting...\033[0m")
        return None

def get_mesh_params(mesh_params):
    # Assertion for mesh parameters: all keys are present; if not, AssertionError with message raised
    expected_keys = ["Nx", "Ny", "Nz", "Lx", "Ly", "Lz"]

    if all(key in mesh_params for key in expected_keys):
        Nx = mesh_params["Nx"]
        Ny = mesh_params["Ny"]
        Nz = mesh_params["Nz"]
        Lx = mesh_params["Lx"]
        Ly = mesh_params["Ly"]
        Lz = mesh_params["Lz"]
        return Nx, Ny, Nz, Lx, Ly, Lz
    else:
        print(f"\033[91m[ERROR - mesh]\033[0m Missing keys: {[key for key in expected_keys if key not in mesh_params]}.\n\033[91mAborting...\033[0m")
        return None
    
def mesh(points, physical_params, mesh_params, t_max=0.1, buffer=20, file_path=None, plot=False, do_gravity=False, data_path=None):
    
    dt, lamb, mu, rho, gamma, default_normal_stress = get_physical_params(physical_params)
    Nx, Ny, Nz, Lx, Ly, Lz = get_mesh_params(mesh_params)

    ##################################################################
    #                         LoadFile                               #
    ##################################################################
    if points is None and file_path is not None:
        data = np.load(file_path) # Load the numpyzip file
        points = data['arr_0']
        # add 20 meters buffer to points
        points = points + buffer
        print(f"\033[32m[INFO - make_mesh]\033[0m File {file_path} successfully loaded, with shape {points.shape}.")
    elif points is not None:
        points = points + buffer
        print(f"\033[32m[INFO - make_mesh]\033[0m Points successfully loaded, with shape {points.shape}.")

    else:
        print(f"\033[91m[ERROR - make_mesh]\033[0m No mesh or file_path provided, cannot continue.")
        return None

    alpha2 = (lamb + 2 * mu) / rho # [m²/s²] Pressure wave velocity squared 
    beta2 = mu / rho # [m²/s²] Shear wave velocity squared
    alpha_beta = np.sqrt(alpha2) / np.sqrt(beta2) # [-] Pressure to Shear velocity ratio
    sigma = (alpha2 - 2 * beta2) / (2 * (alpha2 - beta2)) # Poisson ratio typically between 0.25 to 0.30
    poisson = lamb / (2 * (lamb + mu)) # Also Poisson's ratio, should be equal to sigma
    youngs = ((3*lamb + 2*mu) * mu) / (lamb + mu) # [Pa] Young's Modulus
    bulk = lamb + 2/3 * mu # [Pa] Bulk modulus
    print(f'\033[32m[INFO - make_mesh]\033[0m Alpha² = {alpha2:.2f}. beta² = {beta2:.2f} and sigma (poisson) = {sigma:.2f}, alpha/beta ratio = {alpha_beta:.2f}, poisson= {poisson:.2f}, youngs= {youngs/1e9:.2f} GPa, bulk= {bulk/1e9:.2f} GPa')

    np.set_printoptions(precision=2) # When printing np.arrays, only show two decimals for clarity

    # Stability parameter: dt MUST be smaller than min_times
    min_time_beta = 0.3 * (Lx/(Nx-1))/np.sqrt(beta2) # 0.3 is safety factor
    min_time_alpha = 0.3 *(Ly/(Ny-1))/np.sqrt(alpha2) # 0.3 is safety factor
    print(f"\033[32m[INFO - make_mesh]\033[0m Min time (beta, shear waves) {min_time_beta:.5f}, and (alpha, pressure waves) {min_time_alpha:.5f}")


    if dt > min_time_beta or dt > min_time_alpha:
        print(f"\033[33m[WARNING - make_mesh]\033[0m Dt {dt:.5f}s is larger than {min_time_beta:.5f} or than {min_time_alpha:.5f}")

    Nt = int(t_max / dt) # Number of time iterations


    ##################################################################
    #                   Physical arrays setup                        #
    ##################################################################
    '''All arrays are of length Nt, Nx-1, Ny-1, Nz-1. This is because
    the arrays are staggered, therefore they are each -1 in the 
    dimensional space.'''
    # Positional matrices
    u = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [m] X-pos
    v = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [m] Y-pos
    w = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [m] Z-pos

    # Velocity matrices
    udot = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [m/s] X-velocity
    vdot = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [m/s] Y-velocity
    wdot = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [m/s] Z-velocity

    # Stress matrices:
    # Normal stresses = 20 MPa
    # The rocks are under compression hence normal stress
    txx = np.ones((Nt, Nx-1, Ny-1, Nz-1)) * default_normal_stress # [Pa]
    tyy = np.ones((Nt, Nx-1, Ny-1, Nz-1)) * default_normal_stress # [Pa]
    tzz = np.ones((Nt, Nx-1, Ny-1, Nz-1)) * default_normal_stress # [Pa]
    # Shear stresses = 0 MPa
    # By default, the rocks are not being sheared, hence 0 Mpa
    txy = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [Pa]
    txz = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [Pa]
    tyz = np.zeros((Nt, Nx-1, Ny-1, Nz-1)) # [Pa]

    # Density
    rho = np.ones((Nx-1, Ny-1, Nz-1)) * rho

    # Distribute nodes' position from 0 until the max length
    x = np.linspace(0, Lx, Nx-1) # [m]
    y = np.linspace(0, Ly, Ny-1) # [m]
    z = np.linspace(0, Lz, Nz-1) # [m]

    # Precalculate distance between nodes. This is fine for fixed mesh.
    dx = np.diff(x)[0] # [m]
    dy = np.diff(y)[0] # [m]
    dz = np.diff(z)[0] # [m]

    # Make a meshgrid indexed by 'ij,k'
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    tree = KDTree(points)
    distances, indices = tree.query(grid_points)

    distances = distances.reshape(X.shape)
    threshold = 2.5 * np.array([dx, dy, dz]).max(axis=0)
    mask_exists = distances <= threshold

    indices_grid = indices.reshape(X.shape)

    rho[~mask_exists] = 0

    points_valid = np.vstack([X[mask_exists], Y[mask_exists], Z[mask_exists]]).T
    hull = ConvexHull(points_valid)

    mask_indices = np.flatnonzero(mask_exists)
    surface_flat_indices = mask_indices[hull.vertices]
    surface_ijk = np.unravel_index(surface_flat_indices, X.shape)

    surface_points = points_valid[hull.vertices]
    source_index = 19
    source =  tuple([surface_ijk[0][source_index], surface_ijk[1][source_index], surface_ijk[2][source_index]])

    print(f"\033[32m[INFO - make_mesh]\033[0m Source located at index {source[0]}, {source[1]}, {source[2]}")
    rho[source[0],source[1],source[2]] = 2901 #marker
    # print(f"density at source {rho[source[0],source[1],source[2]]}")

    u[:] = X
    v[:] = Y
    w[:] = Z

    if plot:
        fig, ax = plt.subplots(1,1, figsize=(6,6), subplot_kw={'projection': '3d'})
        ax.set_title("Check if mesh fits inside bounding box")
        ax.scatter(X[mask_exists], Y[mask_exists], Z[mask_exists], alpha=0.2, label="Mesh")
        ax.scatter(X[source], Y[source], Z[source], s=100, color='red', label="Point source")

        ax.plot(X[0,:,0], Y[0,:,0], Z[0,:,0],'--r', label="Bounding box")
        ax.plot(X[:,0,0], Y[:,0,0], Z[:,0,0],'--r')
        ax.plot(X[0,0,:], Y[0,0,:], Z[0,0,:],'--r')

        ax.plot(X[-1,:,-1], Y[-1,:,-1], Z[-1,:,-1],'--r')
        ax.plot(X[:,-1,-1], Y[:,-1,-1], Z[:,-1,-1],'--r')
        ax.plot(X[-1,-1,:], Y[-1,-1,:], Z[-1,-1,:],'--r')
        ax.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
        ax.legend()
        ax.axis("equal")
        plt.tight_layout()
        plt.show()

    # coords = [u, v, w]
    dxs = [dx, dy, dz]
    vels = [udot, vdot, wdot]
    stresses = [txx, tyy, tzz, txy, txz, tyz]

    if do_gravity:
        if data_path is not None:
            gravity_file = "gravity_magnitude_002.npz"
            gravity_path = os.path.join(data_path, gravity_file)
            grav_data = np.load(gravity_path) # Load the numpyzip file
            gravity = grav_data['gravity_magnitude']
            print(f"\033[32m[INFO - make_mesh]\033[0m File {gravity_file} successfully loaded, with shape {gravity.shape}.")
        else:
            print(f"\033[91m[ERROR - make_mesh]\033[0m No gravity_file or data_path provided, cannot continue gravity.")
        

        tolerance = 0.8e-6
        cg_index = np.where(np.isclose(gravity, 0, atol=tolerance))
        print(f"\033[32m[INFO - mesh_plotting]\033[0m CG located at index {cg_index}")
        cg_coords = np.array([x[cg_index[0]], y[cg_index[1]], z[cg_index[2]]]).T
        # print(f"cg coords {cg_coords}")
        voxels = np.stack([X,Y,Z], axis=1)
        # print(f"voxelsshapes {points.shape}")
        print(f"Gridpoints shape {grid_points.shape}, X.shape {X.shape}")
        grid_points = grid_points.reshape(-1,3)
        print(f"gridpoints.reshape {grid_points.shape}")
        # grid_points[~mask_exists] = np.nan
        distances_to_cg = np.linalg.norm(grid_points - cg_coords, axis=1)
        print(f"{distances_to_cg.max()}")
        # problem is because i need to get rid of distance outside the mask such that depth at surface =0
        depth = abs(distances_to_cg - distances_to_cg.max())
        print(f"Depths max {depth.max()}, depth min {depth.min()}")
        print(depth)
        print(f"grid_points {grid_points.shape}")
        print(f"xhspae {X.shape}")
        depth_matrix = depth.reshape(X.shape)
        print(f"Depth matrix {depth_matrix.shape}")
        depth_matrix[~mask_exists] = 0
        print(f"Depths max {np.max(depth_matrix)}, depth min {np.min(depth_matrix)}")
        gravity = gravity[:-1,:-1,:-1] # remove last row column..
        print(f"gravity shape {gravity.shape}")
        print(f"rho.shape {rho.shape}")
        # print(f"X.shape masked {X[mask_exists].shape}")
        pressure_matrix = rho * gravity * depth_matrix
        print(f"pressure.shape {pressure_matrix.shape}")
        print(f"pressure max {np.max(pressure_matrix)}, min {np.min(pressure_matrix)}")
        figp, axp = plt.subplots(1,3,figsize=(6,6))
        surf = axp[0].imshow(pressure_matrix[:,:,int(Nz/2)])
        surf2 = axp[1].imshow(depth_matrix[:,:,int(Nz/2)])
        surf3 = axp[2].imshow(gravity[:,:,int(Nz/2)])
        cbar = figp.colorbar(mappable=surf, ax=axp)
        plt.show()

    return dxs, vels, stresses, rho, source, Nt, gamma, mask_exists


##################################################################
#                          Solver                                #
##################################################################

def solver(dxs, vels, stresses, rho, source, Nt, gamma, physical_params, mesh_params, t_source=20):
    '''The solver loops through all times-1 (Nt-1). 

    If the time is less than 10, it introduces a perturbation at 
    location pos[0:3].

    It then loops through all i,j,k finding the stress at [t+1, i,j,k]
    based on the velocities at time [t, i,j,k].

    With the new stresses, the new velocities at [t+1, i,j,k] through 
    separate for loops. For loop separation is critical such that all
    used stresses are calculated. Important for staggered grid setup.

    With the new velocities, the next new stresses can be calculated,
    and so on.

    All of this is stored in the RAM until it is saved in the savefile
    function.
    '''
    # dt = physical_params[0]
    # lamb = physical_params[1]
    # mu = physical_params[2]
    # # rho = physical_params[3]
    # gamma = physical_params[4]
    # Nx = mesh_params[0]
    # Ny = mesh_params[1]
    # Nz = mesh_params[2]
    # Lx = mesh_params[3]
    # Ly = mesh_params[4]
    # Lz = mesh_params[5]
    dt, lamb, mu, __, gamma, __ = get_physical_params(physical_params)
    Nx, Ny, Nz, __, __, __ = get_mesh_params(mesh_params)
    dx, dy, dz = dxs[0], dxs[1], dxs[2]
    udot, vdot, wdot = vels[0], vels[1], vels[2]
    txx, tyy, tzz, txy, txz, tyz = stresses[0], stresses[1], stresses[2], stresses[3], stresses[4], stresses[5]
    dt = dt
    # t_source = 20
    for t in range(0,Nt-1):
        if t % 20 == 0:
            print(f"\033[32m[INFO - make_mesh]\033[0m t={t*dt:.4f}s, still busy...")

        if t < t_source:
                print(f"\033[32m[INFO - make_mesh: PERTURBATION]\033[0m Displacement applied t= {t}")
                # Pos[[],[],[]] is position of source
                # pos = [np.array([int((Nx/2))-1]), np.array([int(Ny/2)-1]), np.array([int(Nz/2)-1])]
                # pos = [np.array([5,-6]), np.array([int(Ny/2)-1,int(Ny/2)-1]), np.array([int(Nz/2)-1, int(Nz/2)-1])]
                # pos = [np.array([5,-6, int(Nx/2)-1]), np.array([int(Ny/2)-1,int(Ny/2)-1,int(Ny/2)-1]), np.array([int(Nz/2)-1, int(Nz/2)-1, int(Nz/2)-1])]
                # pos = [[1], [1], [1]] # in the bottom left corner
                pos = [source[0], source[1], source[2]]
                # print(f"pos[0] = {pos[0]}")

                udot[t, pos[0], pos[1], pos[2]] = np.sin(np.pi*(t/t_source))
                # vdot[t, pos[0], pos[1], pos[2]] = np.sin(np.pi*(t/t_source))
                # wdot[t, pos[0], pos[1], pos[2]] = np.sin(np.pi*(t/t_source))
                # print(f"density at pos {rho[tuple(pos)]}, udot {udot[t, pos[0], pos[1], pos[2]]}")
                # print(f"next udot {udot[t, pos[0]+1, pos[1], pos[2]]}, {udot[t, pos[0], pos[1]+1, pos[2]]}")

        # Stress calculation loop:
        for k in range(1, Nz-2): # Z-axis

            for j in range(1, Ny-2): # Y-axis

                for i in range(1, Nx-2): # X-axis

                    txx[t+1, i,j,k] = txx[t, i,j,k] + dt * ((lamb + 2*mu) * ((udot[t, i+1,j,k] - udot[t, i,j,k])/dx) + lamb*((vdot[t, i,j,k] - vdot[t, i,j-1,k])/dy + (wdot[t, i,j,k] - wdot[t, i,j,k-1])/dz))
                    tyy[t+1, i,j,k] = tyy[t, i,j,k] + dt * ((lamb + 2*mu) * ((vdot[t, i,j,k] - vdot[t, i,j-1,k])/dy) + lamb*((udot[t, i+1,j,k] - udot[t, i,j,k])/dx + (wdot[t, i,j,k] - wdot[t, i,j,k-1])/dz))
                    tzz[t+1, i,j,k] = tzz[t, i,j,k] + dt * ((lamb + 2*mu) * ((wdot[t, i,j,k] - wdot[t, i,j,k-1])/dz) + lamb*((udot[t, i+1,j,k] - udot[t, i,j,k])/dx + (vdot[t, i,j,k] - vdot[t, i,j-1,k])/dy)) #ok
                    
                    txy[t+1, i,j,k] = txy[t, i,j,k] + dt * mu * ((udot[t, i,j+1,k] - udot[t, i,j,k])/dy + (vdot[t, i,j,k] - vdot[t, i-1,j,k])/dx)
                    txz[t+1, i,j,k] = txz[t, i,j,k] + dt * mu * ((udot[t, i,j,k+1] - udot[t, i,j,k])/dz + (wdot[t, i,j,k] - wdot[t, i-1,j,k])/dx) #ok
                    tyz[t+1, i,j,k] = tyz[t, i,j,k] + dt * mu * ((vdot[t, i,j,k+1] - vdot[t, i,j,k])/dz + (wdot[t, i,j+1,k] - wdot[t, i,j,k])/dy) #ok
        # print(f"[INFO] Stress loop {t} done, starting velocity loop")
        # Velocity calculation loop:
        for k in range(1, Nz-2): # Z-axis

            for j in range(1, Ny-2): # Y-axis

                for i in range(1, Nx-2): # X-axis
                    if rho[i,j,k] == 0:
                        udot[t+1, i,j,k] = 0
                        vdot[t+1, i,j,k] = 0
                        wdot[t+1, i,j,k] = 0
                        # print(f"t={t}, rho[{i},{j},{k}] = 0,thus udot={udot[t+1, i,j,k]}, vdot={vdot[t+1, i,j,k]}, wdot={udot[t+1, i,j,k]}\n")

                    else:
                        # print(f"i{i}, j{j}, k{k}")
                        udot[t+1, i,j,k] = udot[t, i,j,k] * (1 - gamma) + (dt/rho[i,j,k]) * ((txx[t+1, i,j,k] - txx[t+1, i-1,j,k])/dx + (txy[t+1, i,j,k] - txy[t+1, i,j-1,k])/dy + (txz[t+1, i,j,k] - txz[t+1, i,j,k-1])/dz) 
                        vdot[t+1, i,j,k] = vdot[t, i,j,k] * (1 - gamma) + (dt/rho[i,j,k]) * ((txy[t+1, i+1,j,k] - txy[t+1, i,j,k])/dx + (tyy[t+1, i,j+1,k] - tyy[t+1, i,j,k])/dy + (tyz[t+1, i,j,k] - tyz[t+1, i,j,k-1])/dz)
                        wdot[t+1, i,j,k] = wdot[t, i,j,k] * (1 - gamma) + (dt/rho[i,j,k]) * ((txz[t+1, i+1,j,k] - txz[t+1, i,j,k])/dx + (tyz[t+1, i,j,k] - tyz[t+1, i,j-1,k])/dy + (tzz[t+1, i,j,k+1] - tzz[t+1, i,j,k])/dz)
    vels = [udot, vdot, wdot]
    return vels


if __name__ == "__main__":
    # Physical parameters:
    mesh_params = {
    "Nx": int(30), # number of mesh nodes on the x-,y- & z-axis
    "Ny": int(30), # [m] Physical dimension, max length in m of mesh. Needs to be greater than dim
    "Nz": int(30),
    "Lx": int(170),
    "Ly": int(170),
    "Lz": int(170)
    }

    # Physical parameters:
    physical_params = {
        "dt":                    0.0001,    # [s] Time step
        "lamb":                  3.1050e10, # [Pa] Lamé parameter
        "mu":                    3.3075e10, # [Pa] Shear modulus
        "rho":                   600,       # [kg/m³] Density
        "gamma":                 0.0,     # [1/s] Dissipation factor, typical: 0.05 < gamma < 0.1
        "default_normal_stress": 5e6,      # [MPa] Default normal stress put throughout the model
        }
    
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data_folder")
    voxel_file = "voxel_sphere_01_n2308_001.npz"
    voxel_path = os.path.join(data_path, voxel_file)
    
    mesh(
        points=None,
        physical_params=physical_params,
        mesh_params=mesh_params,
        t_max=0.1,
        buffer=25,
        file_path=voxel_path,
        plot=True,
        do_gravity=True,
        data_path=data_path,
        )
