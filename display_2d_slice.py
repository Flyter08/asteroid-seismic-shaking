import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
from IPython.display import HTML
mpl.rcParams['animation.embed_limit'] = 50  # Increase limit to 50 MB, for example

'''This script is made to load save numpyzip files of arrays udot, vdot, wdot and plot them'''

# def get_data(model_name, vels, physical_params, target_axis=0, target_index=None, colormap='terrain', interp='gaussian', save_file=False):
#     '''Need to add a way to read from file if vels is None
#     This function should just unpack data and prepare for plot'''
#     udot, vdot, wdot = vels[0], vels[1], vels[2]
#     dt = physical_params[0]

    

#     # Extract -x,-y,-z values from data from the savefile
#     # udot, vdot, wdot = data['arr_0'], data['arr_1'], data['arr_2'] # velocities

#     # Find the shape of the data in savefile
#     Nx, Ny, Nz, Nt = udot.shape[1], vdot.shape[2], wdot.shape[3], udot.shape[0]
#     # dt = 0.0001 # Time step

#     # 3D geometric sum for velocities at each node (this is for init)
#     # target_axis = 1 # x = 0, y = 1, z = 2
#     if target_index is None:
#         target_index = int(Nz/2) #int(Nz/2)#1 #int(Nz/2) # Plane index number to plot
#     indexer = [slice(None), slice(None), slice(None)] # Create slice objects to slice through 3D x-y-z
#     indexer[target_axis] = target_index # Select the plane of interest along the axis of choice

#     indexer_full = [slice(None)] + indexer # Look through all times
#     indexer_t0 = [0] + indexer # indexer for t=0
#     velocity_magnitude = np.linalg.norm(np.stack([udot[tuple(indexer_t0)],
#                                                 vdot[tuple(indexer_t0)],
#                                                 wdot[tuple(indexer_t0)]],
#                                                 axis=0), axis=0)

#     # print(f"[DEBUG] index full {tuple(indexer_full)}")
#     # Stores all the velocities in x-,y-,z-direction on a centred plane
#     velocities = [udot[tuple(indexer_full)], vdot[tuple(indexer_full)], wdot[tuple(indexer_full)]]

#     # Find the minimum and maximum velocities for scaling the colourbars
#     minmax = [[np.min(velocities[0][:]), np.max(velocities[0][:])],
#             [np.min(velocities[1][:]), np.max(velocities[1][:])],
#             [np.min(velocities[2][:]), np.max(velocities[2][:])]]
    
#     max_values = list(np.max(voxel_centres, axis=0))
#     # min_values = list(np.min(voxel_centres, axis=0))

#     # Magnitude colourbar scaling
#     mag_max = max([minmax[0][1], minmax[1][1], minmax[2][1]])
#     print(f"velocities max {np.max(velocities[0][:])}")
#     print(f"[DEBUG] mag max {mag_max}")
#     # Set the labels and plot titles iteratively
#     xlabels = ['x-nodes [#]','x-nodes [#]','x-nodes [#]']
#     ylabels = ['y-nodes [#]','y-nodes [#]','z-nodes [#]']
#     subtitles = [r'$\dot u$ [m/s]', r'$\dot v$ [m/s]', r'$\dot w$ [m/s]']

#     vel_plots = [] # Initiates vel_plots to store imageAxes for animation function
#     for i, axi in enumerate(ax.flat[:3]): # Loops through the 3 axis x,y,z for the 2D plots
#         min_val, max_val = minmax[i][0], minmax[i][1] # Colour map scaling
#         # imShow init, symetrical log scales, with interpolation method 'gaussian'
#         vel_plot = axi.imshow(velocities[i][0], cmap=colormap, origin='lower',
#                             norm=SymLogNorm(linthresh=0.00001, linscale=0.00001,
#                                             vmin=min_val, vmax=max_val, base=10), interpolation=interp)
#         colorbar = fig.colorbar(vel_plot, ax=axi, label="[m/s]") # Adds colour bars
#         # Plot setup: x-y-labels, plot title
#         axi.set_xlabel(xlabels[i])
#         axi.set_ylabel(ylabels[i])
#         axi.set_title(subtitles[i])
#         vel_plots.append(vel_plot) # Append imageAxis for animation function


#     # Velocity magnitude plot, logarithmic plot, starting close to 0, interpolation
#     mag_plot = ax[1,1].imshow(velocity_magnitude, cmap=colormap, origin='lower',
#                             norm=LogNorm(vmin=0.001, vmax=mag_max, clip=True), interpolation=interp)
#     fig.colorbar(mag_plot, ax=ax[1,1], label="[m/s]")
#     # Plot setup:
#     ax[1,1].set_xlabel(xlabels[0])
#     ax[1,1].set_ylabel(ylabels[0])
#     ax[1,1].set_title(r"Vel. mag. 3D [m/s]")

#     # suptitle_text = fig.suptitle(f"file: {file_name}") # Sets title to figure with filename
#     suptitle_text = plt.suptitle(t='', fontsize= 12)
# def update(frame):
#     '''Animation function:
#     Takes frame as index variable used to loop through all 'time' instances of velocities
#     Updates the imageAxes at each loop call with set_data

#     Returns:
#     result (matplotlib axes handles)
#     This is important for when blit == True.
#     blit == True optimises plotting by only updating necessary parts of plots.
#     '''
#     indexer_frame = [frame+1] + indexer
#     velocity_magnitude = np.linalg.norm(np.stack([udot[tuple(indexer_frame)],
#                                                 vdot[tuple(indexer_frame)],
#                                                 wdot[tuple(indexer_frame)]],
#                                                 axis=0), axis=0)
#     suptitle_text.set_text('file:{} \n t={:.4f}'.format(file_name,frame*dt))
#     # ax[0,0].set_title(rf'$\dot u$ [m/s] t={frame*dt:.4f}')
#     for i, vel_plot in enumerate(vel_plots):
#         vel_plot.set_data(velocities[i][frame,:,:])
            
#     mag_plot.set_data(velocity_magnitude)

#     result = vel_plots + [mag_plot]
#     return result

    # Play animation: On fig, applies function update with time 0:Nt, with refresh every 1ms, 
    # blit=True for faster rendering
    # frames_to_display = np.arange(50, int(Nt/2), 4)
    # ani = FuncAnimation(fig, update, frames=frames_to_display, interval=1, blit=False, cache_frame_data=True)
    # plt.tight_layout() # Arrange plots to avoid overlaps

    # # To save the animation using Pillow as a gif
    # if save_file:
    #     print(f"[INFO] Saving file with name '{file_name}.gif'.")
    #     writer = PillowWriter(fps=5, metadata=dict(artist='Maxime Larguet Sept. 2025'), bitrate=2000)
    #     ani.save(f'{file_name}_002.gif', writer=writer)
    #     print(f"[INFO] File saved successfully.")


    # plt.show()

def update_vel(frame, vels, indexer, vel_plots):
    '''Animation function:
    Takes frame as index variable used to loop through all 'time' instances of velocities
    Updates the imageAxes at each loop call with set_data

    Returns:
    result (matplotlib axes handles)
    This is important for when blit == True.
    blit == True optimises plotting by only updating necessary parts of plots.
    '''
    indexer_frame = [frame] + indexer # indexer for t=0
    velocities_frame = get_vel(indexer_frame, vels)
    # print(f"[INFO - updata_vel] Velocities.shape: {velocities_frame[1].shape}")

    # suptitle_text.set_text('file:{} \n t={:.4f}'.format(model_name, frame*dt))
    # ax[0,0].set_title(rf'$\dot u$ [m/s] t={frame*dt:.4f}')
    for i, vel_plot in enumerate(vel_plots):
        vel_plot.set_data(velocities_frame[i])

    result = vel_plots
    return result

def get_vel(indexer_frame, vels):
    # Simplify? -unnecessary?
    # print(f"[INFO = get_vel] indexer_frame: {indexer_frame}")
    udot, vdot, wdot = vels[0], vels[1], vels[2]
    velocities = [udot[tuple(indexer_frame)], vdot[tuple(indexer_frame)], wdot[tuple(indexer_frame)]]
    return velocities

def get_mag(indexer_frame, vels):
    velocities = get_vel(indexer_frame, vels)
    velocity_magnitude = np.linalg.norm(np.stack([velocities[0],
                                                velocities[1],
                                                velocities[2]],
                                                axis=0), axis=0)

    return velocity_magnitude

def vel_slice(vels, frames_to_show=np.arange(1,100,2), target_axis=0, target_index=None, mag_plot=False, colormap='terrain', interp='gaussian'):
    Nz = vels[2].shape[3]
    if target_index is None:
        target_index = int(Nz/2) #int(Nz/2)#1 #int(Nz/2) # Plane index number to plot
    indexer = [slice(None), slice(None), slice(None)] # Create slice objects to slice through 3D x-y-z
    indexer[target_axis] = target_index # Select the plane of interest along the axis of choice
    indexer_t0 = [0] + indexer # indexer for t=0

    if mag_plot:
        fig, ax = plt.subplots(2,2, figsize=(6,6)) # 4 subplots: x-, y-, z- vels of slice + magnitude plot
    else:
        # Plot setup
        fig, ax = plt.subplots(1,3, figsize=(18,6)) # 3 subplots: x-, y-, z- vels of slice
        xlabels = ['x-nodes [#]','x-nodes [#]','x-nodes [#]']
        ylabels = ['y-nodes [#]','y-nodes [#]','z-nodes [#]']
        subtitles = [r'$\dot u$ [m/s]', r'$\dot v$ [m/s]', r'$\dot w$ [m/s]']

        minmax = [[np.min(vels[0][:]), np.max(vels[0][:])],
            [np.min(vels[1][:]), np.max(vels[1][:])],
            [np.min(vels[2][:]), np.max(vels[2][:])]]
        velocities_t0 = get_vel(indexer_t0, vels)
        vel_plots = [] # Initiates vel_plots to store imageAxes for animation function
        for i, axi in enumerate(ax.flat): # Loops through the 3 axis x,y,z for the 2D plots

            min_val, max_val = minmax[i][0], minmax[i][1] # Colour map scaling

            vel_plot = axi.imshow(velocities_t0[i], cmap=colormap, origin='lower',
                                norm=SymLogNorm(linthresh=0.00001, linscale=0.00001,
                                                vmin=float(min_val), vmax=float(max_val), base=10), interpolation=interp)
            colorbar = fig.colorbar(vel_plot, ax=axi, label="[m/s]") # Adds colour bars
            # Plot setup: x-y-labels, plot title
            axi.set_xlabel(xlabels[i])
            axi.set_ylabel(ylabels[i])
            axi.set_title(subtitles[i])
            vel_plots.append(vel_plot) # Append imageAxis for animation function
        
        ani = FuncAnimation(fig, update_vel, fargs=(vels, indexer, vel_plots), frames=frames_to_show, interval=10, blit=True, cache_frame_data=True)
        plt.tight_layout() # Arrange plots to avoid overlaps
    return HTML(ani.to_jshtml())

def update_mag(frame, vels, indexer, mag_plots, target_axis, target_index):
    '''Animation function:
    Takes frame as index variable used to loop through all 'time' instances of velocities
    Updates the imageAxes at each loop call with set_data

    Returns:
    result (matplotlib axes handles)
    This is important for when blit == True.
    blit == True optimises plotting by only updating necessary parts of plots.
    '''

    for i, vel_plot in enumerate(mag_plots):
        indexer = [slice(None), slice(None), slice(None)] # Create slice objects to slice through 3D x-y-z
        indexer[target_axis[i]] = target_index
        indexer_frame = [frame] + indexer # indexer 
        mag_frame = get_mag(indexer_frame, vels)
        vel_plot.set_data(mag_frame)

    result = mag_plots
    return result

def mag_slice(vels, frames_to_show=np.arange(1,100,2), target_axis=[0], target_index=None, colormap='terrain', interp='gaussian'):
    Nz = vels[2].shape[3]
    # Plot setup
    fig, ax = plt.subplots(1,len(target_axis), figsize=(6*len(target_axis),6)) # 1-3 subplots: x-, y-, z- mags of slice
    xlabels = ['x-nodes [#]','x-nodes [#]','y-nodes [#]']
    ylabels = ['y-nodes [#]','z-nodes [#]','z-nodes [#]']
    subtitles = [r'$||\bf{\dot u}||$ [m/s]', r'$||\bf{\dot u}||$ [m/s]', r'$||\bf{\dot u}||$ [m/s]']
    if target_index is None:
            target_index = int(Nz/2) #int(Nz/2)#1 #int(Nz/2) # Plane index number to plot

    mag_plots = [] # Initiates vel_plots to store imageAxes for animation function
    for i, axi in enumerate(ax.flat): # Loops through the 3 axis x,y,z for the 2D plots
        
        indexer = [slice(None), slice(None), slice(None)] # Create slice objects to slice through 3D x-y-z
        indexer[target_axis[i]] = target_index # Select the plane of interest along the axis of choice
        indexer_t0 = [250] + indexer # indexer for t=1
        mag_t0 = get_mag(indexer_t0, vels)
        
        min_val, max_val = np.min(mag_t0), np.max(mag_t0)

        mag_plot = axi.imshow(mag_t0, cmap=colormap, origin='lower',
                            norm=LogNorm(vmin=0.001, vmax=max_val, clip=True), interpolation=interp)
        colorbar = fig.colorbar(mag_plot, ax=axi, label="[m/s]") # Adds colour bars
        # Plot setup: x-y-labels, plot title
        axi.set_xlabel(xlabels[i])
        axi.set_ylabel(ylabels[i])
        axi.set_title(subtitles[i])
        mag_plots.append(mag_plot) # Append imageAxis for animation function
    
    ani = FuncAnimation(fig, update_mag, fargs=(vels, indexer, mag_plots, target_axis, target_index), frames=frames_to_show, interval=10, blit=True, cache_frame_data=True)
    plt.tight_layout() # Arrange plots to avoid overlaps
    return HTML(ani.to_jshtml())

if __name__ == "__main__":
    # vel_slice()
    pass

# order of operations here:
# call vel_slice()
# vel_slice calls get_data() need get data???
# vel_slice calls get_vel() and/or get_mag() passing the vels from get_data