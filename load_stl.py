import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

def load_model(model_name, stl_path, data_path, dim, scale_factor=None, voxel_pitch=0.015, save_file=False, plot=False):
    mesh = trimesh.load(stl_path)
    extents = mesh.extents
    print(f"\033[32m[INFO - load_stl]\033[0m .stl starting dimensions {extents}.")
    if scale_factor is None and dim is None:
        voxel_pitch = np.mean(extents)/15
        voxelized = mesh.voxelized(pitch=voxel_pitch)  # pitch = size of each voxel in mesh units
        # voxelized() only creates voxels on the surface
        voxelized = voxelized.fill() # Add voxels inside
        voxel_centres = np.float16(voxelized.points) 
    
    # mesh.show() # Show the mesh in 3D with openGL
    elif scale_factor is not None and dim == None:
        mesh.apply_scale(scale_factor)
        
        voxelized = mesh.voxelized(pitch=voxel_pitch*scale_factor)  # pitch = size of each voxel in mesh units
        # voxelized() only creates voxels on the surface
        voxelized = voxelized.fill() # Add voxels inside
        voxel_centres = np.float16(voxelized.points) # Gets the centre points of each voxel

    elif len(dim) ==  3 :
        voxelized = mesh.voxelized(pitch=voxel_pitch)  # pitch = size of each voxel in mesh units
        # voxelized() only creates voxels on the surface
        voxelized = voxelized.fill() # Add voxels inside

        voxel_centres = np.float16(voxelized.points) # Gets the centre points of each voxel
        max_values = list(np.max(voxel_centres, axis=0))
        # min_values = list(np.min(voxel_centres, axis=0))

        scales = np.zeros((len(dim)))
        for i in range(len(dim)):
            scales[i] = dim[i] / max_values[i]
            voxel_centres[:,i] = voxel_centres[:,i] * scales[i]

        max_values = list(np.max(voxel_centres, axis=0))
        print(f"\033[32m[INFO - load_stl]\033[0m Model's max dimensions x-{max_values[0]} m, y-{max_values[1]}, z-{max_values[2]} m, with {voxel_centres.shape[0]} voxels.")
            
    else:
        print(f"\033[31m[ERROR - load_stl]\033[0m Dim.shape[0] is not 3. Expected 3: x,y and z. Returned None")
        voxel_centres = None
        return voxel_centres
        
    if save_file:
            voxels_data_file = f"voxel_{model_name}_n{voxel_centres.shape[0]}_001.npz"
            voxels_data_path = os.path.join(data_path, voxels_data_file)

            np.savez(voxels_data_path, voxel_centres) # Save the file in numpyzip fileformat

            print(f"\033[32m[INFO - load_stl]\033[0m Files save succesfully at {voxels_data_path}.")

    if plot:
        # voxelized.show()
        fig = plt.figure()
        fig.suptitle(f"Generated mesh from voxelized:\n{model_name}")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(voxel_centres[:,0],voxel_centres[:,1],voxel_centres[:,2])
        ax.axis("equal")
        plt.show()

    return voxel_centres


if __name__ == "__main__":
    # dim = [177, 174, 116] # Dimorphos dimensions

    cwd = os.getcwd()
    model_file = "sphere_01.STL"
    model_path = os.path.join(cwd, "models", model_file)
    data_path = os.path.join(cwd, "data_folder")
    

    load_model(model_file, model_path, data_path=data_path, dim=None, scale_factor=None, save_file=False, plot=True)
