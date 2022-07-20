import struct,os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from general_functions import *
import glob

folder = 'one_component'
# folder = 'S3'
# folder = 'two_component'
header, data = loadgadget(f'./{folder}/snapshots/snapshot_000')
particle_count = data[:,0].size

# System Settings
title = 'One Component'
# title = 'S3'
# title = 'Two Component'
comp = [0] # one_component
# comp = [0,int(particle_count/(20+1))] #two component
# comp = [0,int(particle_count/(2+1))] #SIM3

radius_virial = 10
radius_scale = 1
m = header['mass']
rho_0 = (particle_count * m)/(4 * np.pi * (1/(1+10) -1 +np.log(11))) # fix this at some point
G = 1
color = [plt.cm.Set1(i) for i in range(6)]
rlim = 10
interval = 53

# Scraper Settings
overwrite_files = True
lims = 300
dots = 400
lims = 200
dots = 100

# Scraper starts here
print('Preparing environment...')
if not(os.path.exists(f'./{folder}/animations/snaps')): os.makedirs(f'./{folder}/animations/snaps')
if not(os.path.exists(f'./{folder}/animations/energyspace')): os.makedirs(f'./{folder}/animations/energyspace')
if not(os.path.exists(f'./{folder}/analytics/bound')): os.makedirs(f'./{folder}/analytics/bound')
if not(os.path.exists(f'./{folder}/analytics/energy')): os.makedirs(f'./{folder}/analytics/energy')
if not(os.path.exists(f'./{folder}/analytics/circularity')): os.makedirs(f'./{folder}/analytics/circularity')
if not(os.path.exists(f'./{folder}/analytics/frame')): os.makedirs(f'./{folder}/analytics/frame')
if not(os.path.exists(f'./{folder}/analytics/density')): os.makedirs(f'./{folder}/analytics/density')


sizes = comp
sizes = comp.copy()
sizes.append(particle_count)
sizes = np.array(sizes[1:])-np.array(sizes[:-1])
np.savetxt(f"./{folder}/analytics/component_sizes.txt", sizes, fmt="%s")

snap_list = glob.glob(f'./{folder}/snapshots/snapshot_*')
snap_sortd = sorted(snap_list)
bound_mass = np.zeros((len(snap_list),len(comp)))

print('Beginning scrape...')
for num,snap in enumerate(snap_sortd):
    print(f'Loading snapshot_{snap[-3:]}...')
    header, Numfiles = loadgadget_header(snap)
    header, data = loadgadget(snap)

    if len(comp) > 1: data = data[data[:,0].argsort()]

    if num == 0: bound = data[:,0]
    snap_bound = data[np.isin(data[:,0],bound)]

    if overwrite_files or not(os.path.exists(f'./{folder}/analytics/frame/frame_{snap[-3:]}.txt')):
        print('Calculating frame...')
        position = clustercentre(snap_bound[:,1:4],100,0.9)
        velocity = clustercentre(snap_bound[:,4:7],100,0.9)
        frame = np.append(position,velocity)
        np.savetxt(f"./{folder}/analytics/frame/frame_{snap[-3:]}.txt", frame, fmt="%s")
    else:
        print('Loading frame...')
        frame = np.loadtxt(f"./{folder}/analytics/frame/frame_{snap[-3:]}.txt")

    if num != 0:
        if overwrite_files or not(os.path.exists(f'./{folder}/analytics/bound/bound_{snap[-3:]}.txt')):
            print('Stripping unbounded...')
            data_bound = data.copy()
            data_bound[:,1:] -= frame
            data_bound = remove_unbound(data_bound,G,m)
            data_bound = data_bound[np.sqrt(data_bound[:,1]**2+data_bound[:,2]**2+data_bound[:,3]**2)<=rlim]
            data_bound[:,1:] += frame
            np.savetxt(f"./{folder}/analytics/bound/bound_{snap[-3:]}.txt", bound, fmt="%s")
        else:
            print('Loading bounded...')
            bound = np.loadtxt(f"./{folder}/analytics/bound/bound_{snap[-3:]}.txt")
            data_bound = data[np.isin(data[:,0],bound)]
    else:
        data_bound = data.copy()

    bound = data_bound[:,0]

    if num % interval == 0:
        if overwrite_files or not(os.path.exists(f'./{folder}/analytics/energy/energy_{snap[-3:]}.txt')):
            print('Creating Energy...')
            data_bound[:,1:] -= frame
            data_pos = np.array(np.sqrt(data_bound[:,1]**2+data_bound[:,2]**2+data_bound[:,3]**2))
            data_vel = np.array(np.sqrt(data_bound[:,4]**2+data_bound[:,5]**2+data_bound[:,6]**2))
            data_ene = -(calculate_potential_spherical(data_pos,G,m) + 1/2*data_vel**2)
            data_bound[:,1:] += frame

            np.savetxt(f"./{folder}/analytics/energy/energy_{snap[-3:]}.txt", data_ene, fmt="%s")

        if overwrite_files or not(os.path.exists(f'./{folder}/analytics/circularity/circularity_{snap[-3:]}.txt')):
            print('Creating Circularity...')
            data_bound[:,1:] -= frame
            data_cir = np.zeros(len(bound))
            for i in range(len(data_cir)):
                data_cir[i] = NFW_circ(data_bound[i,1:4],data_bound[i,4:7],rho_0,1.0,G)
            data_bound[:,1:] += frame

            np.savetxt(f"./{folder}/analytics/circularity/circularity_{snap[-3:]}.txt", data_cir, fmt="%s")

    print('Calculating density...')
    data_bound[:,1:] -= frame
    xticks = np.logspace(-1,1,50) # sets up spacing for radius
    yticks = np.logspace(-2,1,20) # sets up spacing for density
    distances = np.sqrt(data_bound[:,1]**2 + data_bound[:,2]**2 + data_bound[:,3]**2) # distance of each particle from center of mass
    data_dist = np.column_stack((bound,distances)) # creates an array with ID, distances

    particles, xedges = np.histogram(data_dist[:,1],bins=xticks)
    part_mass = particles*m # mass of particles in each section of a heatmap

    volumes = 4/3 * np.pi * (xedges[1:]**3 - xedges[:-1]**3) # calculates volume at each radius point
    midpoints = (xedges[:-1]+xedges[1:])/2 # calculates midpoints, purely for graphing

    plt.loglog(midpoints,part_mass/volumes,color='k',label='Total') # plots midpoints (radius) vs density

    # filters particles based on component inclusion
    if len(comp) > 1:
        for i,v in enumerate(comp):
            comp_bound = data_dist[data_dist[:,0] >= v]
            if i < len(comp)-1: comp_bound = comp_bound[comp_bound[:,0] < comp[i+1]]
            comp_part, xedges = np.histogram(comp_bound[:,1],bins=xticks)
            comp_mass = comp_part*m
            plt.loglog(midpoints,comp_mass/volumes,ls='-',color=color[i],label=f'Comp{i+1}')

    plt.suptitle(f'Density Profile, Snapshot_{snap[-3:]}')
    plt.xlabel(f'Radius')
    plt.ylabel(f'Density')
    plt.ylim(10**-8,1)
    plt.legend(loc='upper right',framealpha=1,prop={'size':8})
    if overwrite_files or not(os.path.exists(f'./{folder}/analytics/density/density_{snap[-3:]}')): plt.savefig(f'./{folder}/analytics/density/density_{snap[-3:]}',dpi=dots,transparent=False)
    plt.close('all')

    data_bound[:,1:] += frame
    print('Prepping graphs...')
    count = comp.copy()
    count.append(particle_count)
    order = np.array(count[1:]) - np.array(count[:-1])
    order = 1-(order/particle_count)
    comp_legnd = ['']*len(comp)
    fig, ax1 = plt.subplots(figsize=(6,6))
    fig.suptitle(title,fontsize=14)
    ax1.set_xlim(-lims,lims)
    ax1.set_ylim(-lims,lims)
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_aspect(1)

    print('Creating graphs...')
    for i in range(len(comp)):
        if i+1 == len(comp):
            parts = data[data[:,0] >= comp[i],:]
            compb = bound[bound >= comp[i]]
        else:
            parts = data[data[:,0] < comp[i+1],:]
            compb = bound[bound < comp[i+1]]
        bound_mass[num,i] = len(compb)
        comp_legnd[i] = f'Comp{i+1}'
        ax1.scatter(parts[:,1],parts[:,2],color=color[i],s=1,marker='.',alpha=.9,zorder=order[i])
        center = plt.Circle((frame[0],frame[1]), radius=rlim, color='k', fill=False)
        legnd = comp_legnd.copy()
    legnd.append('Center of Mass')
    ax1.add_patch(center)
    ax1.legend(legnd,loc='lower left',framealpha=1,prop={'size':8})
    if overwrite_files or not(os.path.exists(f'./{folder}/animations/snaps/sim{lims}_{snap[-3:]}')): plt.savefig(f'./{folder}/animations/snaps/sim{lims}_{snap[-3:]}',dpi=dots,transparent=False)
    plt.close('all')

if len(comp) > 1:
    print('Concatinating Masses...')
    total = np.zeros((len(snap_list)))
    for i in range(len(comp)):
        total[:] += bound_mass[:,i]
    bound_mass = np.column_stack((bound_mass,total))

print('Writing Logs...')
if overwrite_files or not(os.path.exists(f'./{folder}/animations/sim{lims}_video.mp4')):
    os.system(f"ffmpeg -framerate 6 -y -i ./{folder}/animations/snaps/sim{lims}_%03d.png ./{folder}/animations/sim{lims}_video.mp4")
if overwrite_files or not(os.path.exists(f'./{folder}/analytics/density_video.mp4')):
    os.system(f"ffmpeg -framerate 6 -y -i ./{folder}/analytics/density/density_%03d.png ./{folder}/analytics/density_video.mp4")
if overwrite_files or not(os.path.exists(f'./{folder}/analytics/bound_mass.txt')):
    np.savetxt(f"./{folder}/analytics/bound_mass.txt", bound_mass, fmt="%s")

if len(comp) > 1:
    for i in range(len(comp)):
        plt.plot(range(len(snap_list)),bound_mass[:,i],color=color[i])
    plt.plot(range(len(snap_list)),bound_mass[:,-1],color='k')
else:
    plt.plot(range(len(snap_list)),bound_mass,color='k')

plt.xlabel('Time')
plt.ylabel('Particle Count')
plt.title('Mass Loss',fontsize=14)
mass_legnd = comp_legnd.copy()
mass_legnd.append('Total Bound')
plt.legend(mass_legnd,loc='upper right',framealpha=1,prop={'size':8})
plt.savefig(f'./{folder}/analytics/mass_loss',dpi=dots,transparent=False)
plt.close('all')
