import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
from IPython.display import HTML
mpl.rcParams['animation.embed_limit'] = 100  # Increase limit to 50 MB, for example

'''This script is made to load save numpyzip files of arrays udot, vdot, wdot and plot them'''

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
            # velocities_t0[i] = np.ma.masked_where(velocities_t0[i] ==0, velocities_t0[i])
            # cmap = plt.cm.RdBu
            # cmap.set_bad('black')

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

    for i, mag_plot in enumerate(mag_plots):
        indexer = [slice(None), slice(None), slice(None)] # Create slice objects to slice through 3D x-y-z
        indexer[target_axis[i]] = target_index
        indexer_frame = [frame] + indexer # indexer 
        mag_frame = get_mag(indexer_frame, vels)
        mag_plot.set_data(mag_frame)

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

        mag_t0_masked = np.ma.masked_where(mag_t0==0.0, mag_t0)
        
        min_val, max_val = np.min(mag_t0_masked), np.max(mag_t0_masked)
        cmap = plt.get_cmap(colormap)
        cmap.set_under(color='white')
        norm = LogNorm(vmin=0.001, vmax=max_val, clip=False)
        mag_plot = axi.imshow(mag_t0_masked, cmap=cmap, origin='lower',
                            norm=norm, interpolation=interp)
        colorbar = fig.colorbar(mag_plot, ax=axi, label="[m/s]") # Adds colour bars
        # Plot setup: x-y-labels, plot title
        axi.set_xlabel(xlabels[i])
        axi.set_ylabel(ylabels[i])
        axi.set_title(subtitles[i])
        mag_plots.append(mag_plot) # Append imageAxis for animation function
    
    ani = FuncAnimation(fig, update_mag, fargs=(vels, indexer, mag_plots, target_axis, target_index), frames=frames_to_show, interval=10, blit=True, cache_frame_data=True)
    # Arrange plots to avoid overlaps
    plt.tight_layout() 

    return HTML(ani.to_jshtml())
    
def get_fft(vels, frames_to_show, target_axis=[0], target_index=None, coords=[None,None]):
    
    if target_index is None:
        Nz = vels[2].shape[3]
        target_index = int(Nz/2) #int(Nz/2)#1 #int(Nz/2) # Plane index number to plot
    if coords is None:
        coords = [int(Nz/2), int(Nz/2)]

    indexer = [slice(None), slice(None), slice(None)] # Create slice objects to slice through 3D x-y-z
    indexer[target_axis[0]] = target_index # Select the plane of interest along the axis of choice


    mags_ = []
    for i in frames_to_show-1:
        indexer_t = [i] + indexer #last frame # STOOPID
        mags_.append(get_mag(indexer_frame=indexer_t, vels=vels)[coords[0], coords[1]])
    # print(mags_)
    fft_mag = np.fft.fft(mags_)
    fft_freq_mag = np.fft.fftfreq(len(mags_), 0.0001)
    fig_fft, axfft = plt.subplots(1, 1, figsize=(6,6)) 
    fft =axfft.semilogy(fft_freq_mag, np.abs(fft_mag))
    axfft.set(xlabel="Frequency [Hz]", ylabel="Amplitude", title=f"Frames:{frames_to_show[0]}-{frames_to_show[-1]}\nCoords {coords[0],coords[1]}")
    axfft.grid("minor")
    from scipy.signal import find_peaks
    # peaks, _ = find_peaks(np.abs(fft_mag), height=1e1, distance=150)
    peaks, _ = find_peaks(np.abs(fft_mag), prominence=5, distance=50)
    print(f"Frequencies peak {np.unique(np.abs(fft_freq_mag[peaks]))} Hz")
    # Draw vertical lines for each detected peak in the Sun's FFT
    for peak in peaks:
        axfft.axvline(x=fft_freq_mag[peak], color="red", linestyle="--", label="Peaks")

    # instead lets plot the magnitude of that singular point over time
    # axfft.plot(frame_to_display*0.0001, mags_)


def get_fft_slice(vels, frames_to_show, target_axis=[0], target_index=None):
    from scipy.signal import find_peaks # Use scipy's signal analysis for peaks

    if target_index is None:
        Nz = vels[2].shape[3]
        target_index = int(Nz/2) #int(Nz/2)#1 #int(Nz/2) # Plane index number to plot
    
    colormap="plasma"
    cmap = plt.get_cmap(colormap).copy()
    fig_fft, axfft = plt.subplots(1, len(target_axis), figsize=(4*len(target_axis),4))
    fig_fft.suptitle(f"FFT of slice(s) through node <{Nz}> and frames <{frames_to_show[0]}-{frames_to_show[-1]}>: Dominant peaks")

    for k in range(len(target_axis)):
        print(f"\033[32m[INFO - get_fft_slice]\033[0m Working axis {k}")
        indexer = [slice(None), slice(None), slice(None)] # Create slice objects to slice through 3D x-y-z
        indexer[target_axis[k]] = target_index # Select the plane of interest along the axis of choice


        mags_ = []
        peaks_freq = np.empty([Nz, Nz])
        for t in frames_to_show-1: # loop through all times
            indexer_t = [t] + indexer
            mags_.append(get_mag(indexer_frame=indexer_t, vels=vels)) # Get all magnitudes of all times and all nodes
        mags_ = np.array(mags_)
        # print(f"mags_ shape {mags_.shape}")
        # print(f"peaks_freq.shape {peaks_freq.shape}")
            

        for i in range(Nz): # do ffts for each node
            for j in range(Nz):
                fft_mag = np.fft.fft(mags_[:,i,j])
                fft_freq_mag = np.fft.fftfreq(len(mags_[:,i,j]), 0.0001)
                peaks, _ = find_peaks(np.abs(fft_mag), prominence=5, distance=50)
                # print(f"{i},{j} peak freq {np.unique(np.abs(fft_freq_mag[peaks]))}")
                try:
                    peaks_freq[i,j] = np.unique(np.abs(fft_freq_mag[peaks]))[-1]
                except:
                    peaks_freq[i,j] = np.nan
                # print(f"{i},{j} peak freq {peaks_freq[i,j]}")
                
                
         
        fft = axfft[k].imshow(peaks_freq, origin="lower", cmap=cmap, vmin=np.nanmin(peaks_freq), vmax=np.nanmax(peaks_freq))
        axfft[k].set_title(f"Axis {target_axis[k]}")
        cbar = fig_fft.colorbar(fft, ax=axfft[k], shrink=1)
    plt.tight_layout
    


if __name__ == "__main__":

    axis_to_display = [0, 1, 2] # 0 = x, 1 = y, 2 = z
    start_frame = 1
    end_frame = 5000#Nt#int(Nt/2)
    interval = 1
    frame_to_display = np.arange(start_frame, end_frame, interval)
    cwd = os.getcwd()
    velocity_file = f"vels_sphere_01_n2308_002.npz"
    data_path = os.path.join(cwd, "data_folder")
    velocity_path = os.path.join(data_path, velocity_file)
    data = np.load(velocity_path)
    vels = data["arr_0"]
    test_coords = [[15,3],[10,15], [15,15], [20,15]]
    # for i in range(len(test_coords)):
        # get_fft(vels, frame_to_display, axis_to_display, target_index=None, coords=test_coords[i])
    get_fft_slice(vels, frame_to_display, axis_to_display, target_index=None)
    plt.show()