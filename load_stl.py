import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

def load_model(model_name, file_path, dim, voxel_pitch=0.015, save_file=False, plot=False):
    if len(dim) ==  3 :
        mesh = trimesh.load(file_path)
        # mesh.show() # ABSOLUTE BANGER

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
        print(f"[INFO - load_stl] Model's max dimensions x-{max_values[0]} m, y-{max_values[1]}, z-{max_values[2]} m, with {voxel_centres.shape[0]} voxels.")
        


        ##################################################################
        #                         Savefile                               #
        ##################################################################

        if save_file:
            # folder = "3D models\data_folder" # Savefile location : root\data_folder
            data_folder = "data_folder"
            parent_dir = os.path.dirname(file_path) # ..\3D models\v2\models
            parent_dir = os.path.dirname(parent_dir) # ..\3D models\v2
            new_path = os.path.join(parent_dir, data_folder) # ..\3D models\v2\data_folder
            file_path_save = os.path.join(new_path, f"voxel_{model_name}_n{voxel_centres.shape[0]}_001.npz") # Save filename.
            np.savez(file_path_save, voxel_centres) # Save the file in numpyzip fileformat
            print(f"[INFO - load_stl] Files save succesfully at {file_path_save}.")

        if plot:
            # voxelized.show()
            fig = plt.figure()
            fig.suptitle(f"Generated mesh from voxelized:\n{model_name}")
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(voxel_centres[:,0],voxel_centres[:,1],voxel_centres[:,2])
            plt.show()
    else:
        print(f"[ERROR - load_stl] Dim.shape[0] is not 3. Expected 3: x,y and z. Returned None")
        voxel_centres = None

    return voxel_centres


if __name__ == "__main__":
    folder = "3D models\\v2\\models"
    file_name = "dimorphos_decimated_2.5k.stl"
    file_path = os.path.join(folder, file_name)
    dim = [177, 174, 116] # Dimorphos dimensions
    load_model(file_name, file_path, dim, save_file=False, plot=True)
