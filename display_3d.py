import os
import time
import imageio
import numpy as np
import open3d as o3d
from matplotlib import colormaps as cm

def show3d(file_path, mask_exists, plot_pcd=False, plot_voxels=True, save_file=False, colormap='terrain'):

    # Data source
    cwd = os.getcwd()
    image_folder = "images"
    # image_folder = "3D models\\v2\\images"
    save_image_name = "voxel_image_001.gif"
    save_image_path = os.path.join(cwd, image_folder, save_image_name)
    data = np.load(file_path)

    # Extract -x,-y,-z values from data from the savefile
    vels = data['arr_0']
    udot, vdot, wdot = vels[0], vels[1], vels[2]
    Lx, Ly, Lz = 220, 220, 220
    # Find the shape of the data in savefile
    Nx, Ny, Nz, Nt = udot.shape[1], vdot.shape[2], wdot.shape[3], udot.shape[0]
    X, Y, Z = np.meshgrid(np.linspace(0,Lx, Nx), np.linspace(0, Ly, Ny), np.linspace(0, Lz, Nz), indexing='ij')
    dx = np.diff(X,axis=0)[0,0,0]
    # print(f"DX {dx}")

    mesh_vertices = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    mask_flat = mask_exists.ravel()
    print(f"\033[32m[INFO - show3d]\033[0m Flattened mask shape {mask_flat.shape[0]}")
    filtered_vertices = mesh_vertices[mask_flat]

    dt = 0.0001 # Time step

    # Stores all the velocities in x-,y-,z-direction on a centred plane
    velocities = [udot[:, :,:,:], vdot[:, :,:,:], wdot[:, :,:,:]]

    ### INIT###
    velocities_magnitude = np.linalg.norm(np.stack([velocities[0][1],
                                                velocities[1][1],
                                                velocities[2][1]],
                                                axis=0), axis=0)

    velocities_magnitude_masked = velocities_magnitude[mask_exists]

    # Normalize masked magnitudes only
    vmin, vmax = velocities_magnitude_masked.min(), velocities_magnitude_masked.max()
    normed = (velocities_magnitude_masked - vmin) / (vmax - vmin)

    # Use matplotlib colormap
    cmap = cm.get_cmap('viridis')
    colors_rgba = cmap(normed)  # shape (N, 4), floats between 0-1

    # Convert to uint8 [0-255] for trimesh colors
    colors_uint8 = (colors_rgba[:, :4] * 255).astype(np.uint8)
    colors = colors_rgba[:, :3]  # drop alpha channel
    # print(f"colors{colors.shape}")

    # Create point cloud and assign positions and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize point cloud with velocity-based coloring
    if plot_pcd:
        print("\033[32m[INFO - show3d]\033[0m Displaying colored point cloud...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Velocity Colored Point Cloud")
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 50
    # vis.run()

    if plot_voxels:
        voxel_size = dx + 0.001#2.87 # Choose voxel size as appropriate for your mesh scale
        # Create voxel grid from points
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        print("\033[32m[INFO - show3d]\033[0m Displaying colored voxel grid...")
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(window_name="Velocity Colored Voxel Cloud")
        vis2.add_geometry(voxel_grid)
        print(f"\033[32m[INFO - show3d]\033[0m PCD shape {filtered_vertices.shape[0]}, Voxel {np.array(voxel_grid.get_voxels()).shape[0]}")

    if plot_pcd or plot_voxels:
        max_frame = Nt # Nt
        images = []
        for frame in range(1,max_frame):
            velocities_magnitude = np.linalg.norm(np.stack([velocities[0][frame],
                                                    velocities[1][frame],
                                                    velocities[2][frame]],
                                                    axis=0), axis=0)

            velocities_magnitude_masked = velocities_magnitude[mask_exists]

            # Normalize masked magnitudes only
            vmin, vmax = velocities_magnitude_masked.min(), velocities_magnitude_masked.max()
            normed = (velocities_magnitude_masked - vmin) / (vmax - vmin)

            # Use matplotlib colormap
            cmap = cm.get_cmap(colormap)
            colors_rgba = cmap(normed)[:, :3]  # shape (N, 4), floats between 0-1
            pcd.colors = o3d.utility.Vector3dVector(colors_rgba)
            
            # UPDATE POINT CLOUD
            if plot_pcd:
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            # UPDATE VOXEL GRID
            if plot_voxels:
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

                vis2.clear_geometries()
                vis2.add_geometry(voxel_grid, reset_bounding_box=False)

                vis2.poll_events()
                vis2.update_renderer()
            if save_file and frame % 10 == 0: # within time loop for gif creation
                if vis2 is not None:
                    image = vis2.capture_screen_float_buffer(do_render=True)
                    image_uint8 = (np.asarray(image)*255).astype(np.uint8)
                    images.append(image_uint8)
            time.sleep(0.01)

        
        # Kill window after finishing ~
        if plot_pcd:
            vis.destroy_window()
        if plot_voxels:
            vis2.destroy_window()

        if save_file and plot_voxels:
            imageio.mimsave(save_image_path, images, fps=5)
    else:
        print(f"\033[31m[INFO - show3d]\033[0m No plotting requested . . .")


if __name__ == "__main__":
    pass