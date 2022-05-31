import struct,os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from general_functions import *
import glob

radius_virial = 10
radius_scale = 1
particle_count = 100000
G = 1
size = int(particle_count/(20+1))

snap_list = glob.glob('./snapshots/snapshot_*')
overwrite_files = True
system_plots = True
mass_plot = True
in_frame = False
create_mp4 = True
lims = 300
dots = 400

snap_sorted = sorted(snap_list)
log_bound = np.zeros((len(snap_list),3))
log_frame = np.zeros((len(snap_list),6))

for num,snap in enumerate(snap_sorted):
    header, Numfiles = loadgadget_header(snap)
    header, data = loadgadget(snap)

    data_snap = data[data[:,0].argsort()]
    mass_snap = header['mass']

    if num == 0: data_bound = data_snap.copy()

    ### Find Frame of System ###################################################
    # If files exist with frame offsets in ./analysis_data and if              #
    # overwrite_files is False, they will be used to find the frame.           #
    # Otherwise, files will be created and saved for future reference.         #
    ############################################################################

    # Find frame based on last bound particles
    path_frame = f'./analysis_data/frame_{snap[-3:]}'
    if os.path.exists(path_frame) and not(overwrite_files):
        with open(path_frame) as f:
            frame_string = f.read().splitlines()
            frame_data = np.asarray(frame_string, dtype=np.float64, order='C')
    else:
        data_bound = np.intersect1d(data_snap[:,0],data_bound)
        data_bound = data_snap[np.isin(data_snap[:,0],data_bound)]

        frame_pos = clustercentre(data_bound[:,1:4],1000,0.9)
        frame_vel = clustercentre(data_bound[:,4:],1000,0.9)
        frame_data = np.append(frame_pos,frame_vel)
        with open(f'./analysis_data/frame_{snap[-3:]}', 'w') as f:
            for i in frame_data:
                f.write(f'{i}\n')

    ### Find Bound Particles ####################################################
    # If files exist with bound particles IDs in ./analysis_data and if         #
    # overwrite_files is False, they will be used to find the bound particles.  #
    # Otherwise, files will be created and saved for future reference.          #
    #############################################################################

    path_bound = f'./analysis_data/bound_{snap[-3:]}'
    if os.path.exists(path_bound) and not(overwrite_files):
        with open(path_bound) as f:
            for line in f:
                bound_string = f.read().splitlines()
        bound_array = np.asarray(bound_string, dtype=np.float64, order='C')
        data_bound = data_snap[np.isin(data_snap[:,0],bound_array)]
        data_bound[:,1:] -= frame_data
    else:
        data_bound[:,1:] -= frame_data
        if num > 0: data_bound = remove_unbound(data_bound,G,mass_snap)
        with open(f'./analysis_data/bound_{snap[-3:]}', 'w') as f:
            for i in data_bound[:,0]:
                f.write(f'{int(i)}\n')
    print(f'{num} - {data_bound[:,0].size}')
    data_unbound = data_snap[~np.isin(data_snap[:,0],data_bound[:,0])]
    data_unbound[:,1:] -= frame_data

    log_frame[num,0:] = frame_data
    log_bound[num,0] = data_bound[:,0].size
    log_bound[num,1] = (data_bound[data_bound[:,0] < size,0]).size
    log_bound[num,2] = (data_bound[data_bound[:,0] >= size,0]).size

    ## -- Plot System -- ##
    # If system_plots is True, the script will plot all snapshots of the
    # simulation and save pngs of all plots in ./animations/
    # If in_frame is True, the plots will be in the frame of the system.

    if system_plots:
        if not(in_frame):
            data_bound[:,1:] += frame_data
            data_unbound[:,1:] += frame_data

        bound_one = data_bound[data_bound[:,0] < size]
        bound_one = data_bound[np.isin(data_bound[:,0],bound_one[:,0])]
        unbound_one = data_unbound[data_unbound[:,0] < size]
        unbound_one = data_unbound[np.isin(data_unbound[:,0],unbound_one[:,0])]

        bound_two = data_bound[data_bound[:,0] >= size]
        bound_two = data_bound[np.isin(data_bound[:,0],bound_two[:,0])]
        unbound_two = data_unbound[data_unbound[:,0] >= size]
        unbound_two = data_unbound[np.isin(data_unbound[:,0],unbound_two[:,0])]

        fig, ax1 = plt.subplots(figsize=(6,6))
        fig.suptitle('Two-Component Simulation: 20M1 = M2',fontsize=14)
        ax1.scatter(unbound_one[:,1],unbound_one[:,2],color=plt.cm.Paired(4),s=1,marker='.',alpha=1,zorder=1.5)
        ax1.scatter(bound_one[:,1],bound_one[:,2],color=plt.cm.Paired(5),s=1,marker='o',alpha=1,zorder=2)
        ax1.scatter(unbound_two[:,1],unbound_two[:,2],color=plt.cm.Paired(1),s=1,marker='.',alpha=1,zorder=1)
        ax1.scatter(bound_two[:,1],bound_two[:,2],color=plt.cm.Paired(0),s=1,marker='.',alpha=1,zorder=2.5)
        ax1.plot(frame_data[0],frame_data[1],color='k',marker='+',ms=10,zorder=3.0)
        ax1.set_xlim(-lims,lims)
        ax1.set_ylim(-lims,lims)
        ax1.set_xlabel('X Axis')
        ax1.set_ylabel('Y Axis')
        ax1.legend(['M1 Unbounded','M1 Bounded','M2 Unbounded','M2 Bounded'],loc='lower left',framealpha=1,prop={'size':8})
        ax1.set_aspect(1)
        plt.savefig(f'./animations/sim{lims}_{snap[-3:]}',dpi=dots,transparent=False)
        plt.close('all')

if create_mp4: os.system(f"ffmpeg -framerate 6 -y -i ./animations/sim{lims}_%03d.png sim{lims}.mp4")

## -- Plot Mass -- ##
# If mass_plot is True, the script will plot the mass loss of the simulation.

if mass_plot:
    plt.plot(range(len(snap_list)),log_bound[:,0]*mass_snap,c='k',zorder=1)
    plt.plot(range(len(snap_list)),log_bound[:,1]*mass_snap,c=plt.cm.Paired(5),linestyle='--',zorder=1.5)
    plt.plot(range(len(snap_list)),log_bound[:,2]*mass_snap,c=plt.cm.Paired(1),linestyle='--',zorder=1.5)
    plt.xlabel('Snapshots')
    plt.ylabel('Bounded Mass')
    plt.title('Two-Component Bounded Mass Loss')
    plt.legend(['Total Bound','M1 Bounded','M2 Bounded'])
    plt.savefig(f'./mass_loss',dpi=500,transparent=False)
    plt.close('all')

plt.plot(range(len(snap_list)),log_frame[:,0])
plt.show()