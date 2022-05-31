import struct,os
# from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import integrate
from general_functions import *

num = 0 # file number that you want to analyze
overwrite = True # overwrites old data files
save_plot = True

# =======================================================================
# Open Data Files                                                        
# =======================================================================

# sets up string segment and opens file
if num > 9:
    file = '0' + str(num)
else:
    file = '00' + str(num)

filename = f'./snapshots/snapshot_{file}'
header, Numfiles = loadgadget_header(filename)
header, data = loadgadget(filename)
m = header['mass']
G = 1
rho_0 = (data[:,0].size * m)/(4 * np.pi * (1/(1+10) -1 +np.log(11))) # fix this at some point
scale_radius = 1.0

# finds bound particle data for the snapshot prior to the one opened
with open(f'./analysis_data/bound_{file}') as f:
    bound_string = f.read().splitlines()
    bound_IDs = np.asarray(bound_string, dtype=np.float64, order='C')

with open(f'./analysis_data/frame_{file}') as f:
    frame_string = f.read().splitlines()
    frame = np.asarray(frame_string, dtype=np.float64, order='C')

# uses last snapshot's bound particle data to filter particles from currently opened file 
bound_parts = np.intersect1d(data[:,0],bound_IDs)
data_bound = data[np.isin(data[:,0],bound_parts)]
data_unbound = data[np.isin(data[:,0],bound_parts,invert=True)]
# plt.scatter(data_unbound[:,1],data_unbound[:,2],marker='.')
# plt.scatter(data_bound[:,1],data_bound[:,2],marker='.')
# plt.ylim(-300,300)
# plt.xlim(-300,300)
# plt.show()
# plt.close('all')

# =======================================================================
# Volumes and Masses                                                     
# =======================================================================

# find frame and shift points
data_bound[:,1:] -= frame
data_unbound[:,1:] -= frame
dist_bound = np.sqrt(data_bound[:,1]**2+data_bound[:,2]**2+data_bound[:,3]**2)
dist_unbound = np.sqrt(data_unbound[:,1]**2+data_unbound[:,2]**2+data_unbound[:,3]**2)

# groups particles into bins based on distance from center
edges = np.logspace(-1,1,10)

# # graphs visual for bins
# fig, ax = plt.subplots()
# ax.scatter(data_bound[:,1],data_bound[:,2],marker='.')
# for i in edges:
#     ax.add_patch(plt.Circle((0,0),i,fill=False))
# plt.show()
# plt.close('all')

particles, bins = np.histogram(dist_bound,bins=edges)
particle_mass = particles * m

volumes = np.zeros(len(particles))
dist = (bins[:-1]+bins[1:])/2

# add up all the particles between each bin value
for i in range(len(particles)):
    value, _ = integrate.quad(vol_sphere,bins[i],bins[i+1])
    volumes[i] = value

plt.xticks(bins)
plt.hist(dist_bound,bins=bins)
if save_plot == True: plt.savefig(f'./density_{file}',dpi=400,transparent=False)
# plt.show()
plt.close('all')

# ========================================================================
# Compare NFW Profile                                                    
# ========================================================================
# print('Graphing NFW profiles...')

volumes = 4/3*np.pi* (bins[1:]**3 - bins[:-1]**3)

plt.title('Density')
plt.loglog(dist,(particle_mass/volumes),label='System')
plt.legend()
if save_plot == True: plt.savefig(f'./profile_{file}',dpi=400,transparent=False)
# plt.show()
plt.close('all')

# ========================================================================
# Calculate Momentum and Energy
# ========================================================================
# Opens analytics (*_energy and *_circularity files if they exist. If not,
# generates files and saves them in ./analytics/ for future usage

if overwrite & os.path.exists(f'./analysis_data/energy_{file}'):
    print('Overwriting analytics files...')
    os.remove(f'./analysis_data/energy_{file}')
    os.remove(f'./analysis_data/energy_{file}')

if not(os.path.exists(f'./analysis_data/energy_{file}')):
    print('Creating analytics...')
    momentum = np.cross(data_bound[:,1:4],data_bound[:,4:7])*m
    parts_mom = np.sqrt(momentum[:,0]**2+momentum[:,1]**2+momentum[:,2]**2)

    parts_pos = np.array(np.sqrt(data_bound[:,1]**2+data_bound[:,2]**2+data_bound[:,3]**2))
    parts_vel = np.array(np.sqrt(data_bound[:,4]**2+data_bound[:,5]**2+data_bound[:,6]**2))
    parts_ene = -(calculate_potential_spherical(parts_pos,G,m) + 1/2*parts_vel**2)

    circ = np.zeros(len(data_bound[:,0]))
    for i in range(len(circ)):
        circ[i] = NFW_circ(data_bound[i,1:4],data_bound[i,4:7],rho_0,1.0,G)

    with open(f'./analysis_data/energy_{file}', 'w') as f:
        for i in parts_ene:
            f.write("%s\n" % i)

    with open(f'./analysis_data/circ_{file}', 'w') as f:
        for i in circ:
            f.write("%s\n" % i)
else:
    print('Loading scrapped analytics...')
    with open(f'./analysis_data/energy_{file}') as f:
        data_txt = f.read().splitlines()
        parts_ene = np.asarray(data_txt, dtype=np.float64, order='C')

    with open(f'./analysis_data/circ_{file}') as f:
        data_txt = f.read().splitlines()
        circ = np.asarray(data_txt, dtype=np.float64, order='C')

edges_yaxis = np.linspace(0,np.max(circ),25)
edges_xaxis = np.linspace(0,np.max(parts_ene),25)

heatmap, xedges, yedges = np.histogram2d(parts_ene, circ, bins=(edges_xaxis,edges_yaxis))

plt.clf()
# plt.hist2d(np.log10(parts_ene),np.log10(circ),bins=bin_edges)
plt.imshow(np.log10(heatmap.T), origin='lower',aspect='auto',extent=(edges_xaxis[0],edges_xaxis[-1],edges_yaxis[0],edges_yaxis[-1]))
plt.title(f'Snapshot_{file} - Energy vs Circularity')
plt.xlabel('Energy')
plt.ylabel('L/Lmax')
if save_plot == True: plt.savefig(f'./energy_{file}',dpi=400,transparent=False)
# plt.show()
plt.close('all')

# =======================================================================
# Energy vs Circularity Comparisons
# =======================================================================

filename = f'./snapshots/snapshot_000'
header, Numfiles = loadgadget_header(filename)
header, data = loadgadget(filename)

num_snap = 52
num_orb = 10

all_emax, one_emax, two_emax = 0.04, 0.04, 0.04
all_cmax, one_cmax, two_cmax = 1, 1, 1

radius_virial = 10
radius_scale = 1
particle_count = data[:,0].size
G = 1
size = int(particle_count/(20+1))

hall_ene = np.linspace(0,all_emax,20)
hall_cir = np.linspace(0,all_cmax,20)

hone_ene = np.linspace(0,one_emax,20)
hone_cir = np.linspace(0,one_cmax,20)

htwo_ene = np.linspace(0,two_emax,20)
htwo_cir = np.linspace(0,two_cmax,20)

for i in range(num_snap-1):
    if i != 0:
        if i % num_orb == 0:
            num_file = str(i)
            count_orb = str(int(i/num_orb))

            while len(num_file) < 3: num_file = '0' + num_file

            with open(f'./analysis_data/bound_{num_file}') as f:
                orbit_txt = f.read().splitlines()
                bound_txt = np.asarray(orbit_txt, dtype=np.float64, order='C')

            bound_orb = np.intersect1d(all_ini[:,0],bound_txt)
            all_orb = all_ini[np.isin(all_ini[:,0],bound_orb)]

            one_orb = all_orb[all_orb[:,0] < size]
            one_orb = all_orb[np.isin(all_orb[:,0],one_orb[:,0])]
            two_orb = all_orb[np.isin(all_orb[:,0],one_orb[:,0],invert=True)]

            print(f'--- Orbit {count_orb} ---')
            print(f'Total particle count: {particle_count}')
            print(f'Total bound particle count: {len(all_orb[:,0])}')
            print(f'Component 1 bound particle count: {len(one_orb[:,0])}')
            print(f'Component 2 bound particle count: {len(two_orb[:,0])}')
            print()

            plt.clf()
            heatone_orb, xone_orb, yone_orb = np.histogram2d(one_orb[:,1], one_orb[:,2], bins=(hone_ene,hone_cir))
            heatone_com = heatone_orb/heatall_ini
            heatone_com[np.isnan(heatone_com)] = 0
            plt.imshow(heatone_com.T, origin='lower',aspect=one_emax,cmap='jet',vmin=0,vmax=1,extent=(0,one_emax,0,one_cmax))
            plt.colorbar(label='Fraction of Initially Bound Particles')
            plt.title(f'Orbit {count_orb} - Remainder of Component 1 Particles')
            plt.xlabel('Energy')
            plt.ylabel('L/Lmax')
            if save_plot == True: plt.savefig(f'./orbit_onecomp_{count_orb}',dpi=400,transparent=False)
            # plt.show()
            plt.close('all')

            plt.clf()
            heattwo_orb, xtwo_orb, ytwo_orb = np.histogram2d(two_orb[:,1], two_orb[:,2], bins=(htwo_ene,htwo_cir))
            heattwo_com = heattwo_orb/heatall_ini
            heattwo_com[np.isnan(heattwo_com)] = 0
            plt.imshow(heattwo_com.T, origin='lower',aspect=two_emax,cmap='jet',vmin=0,vmax=1,extent=(0,two_emax,0,two_cmax))
            plt.colorbar(label='Fraction of Initially Bound Particles')
            plt.title(f'Orbit {count_orb} - Remainder of Component 2 Particles')
            plt.xlabel('Energy')
            plt.ylabel('L/Lmax')
            if save_plot == True: plt.savefig(f'./orbit_twocomp_{count_orb}',dpi=400,transparent=False)
            # plt.show()
            plt.close('all')
    else:
        with open(f'./analysis_data/bound_000') as f:
            log_ini = f.read().splitlines()
            bound_ini = np.asarray(log_ini, dtype=np.float64, order='C')

        with open(f'./analysis_data/energy_000') as f:
            log_ini = f.read().splitlines()
            energy_ini = np.asarray(log_ini, dtype=np.float64, order='C')

        with open(f'./analysis_data/circ_000') as f:
            log_ini = f.read().splitlines()
            circ_ini = np.asarray(log_ini, dtype=np.float64, order='C')

        all_ini = np.column_stack((bound_ini,energy_ini,circ_ini))

        one_ini = all_ini[all_ini[:,0] < size]
        one_ini = all_ini[np.isin(all_ini[:,0],one_ini[:,0])]
        two_ini = all_ini[np.isin(all_ini[:,0],one_ini[:,0],invert=True)]

        plt.clf()
        heatall_ini, xall_ini, yall_ini = np.histogram2d(all_ini[:,1], all_ini[:,2], bins=(hall_ene,hall_cir))
        plt.imshow(heatall_ini.T, origin='lower',aspect=all_emax,cmap='jet',extent=(0,all_emax,0,all_cmax),vmax=1,vmin=0)
        # plt.colorbar(label='Number of Bound Particles')
        plt.title(f'Initially Bound Particles')
        plt.xlabel('Energy')
        plt.ylabel('L/Lmax')
        if save_plot == True: plt.savefig(f'./orbit_000',dpi=400,transparent=False)
        # plt.show()
        plt.close('all')

        plt.clf()
        heatone_ini, xone_ini, yone_ini = np.histogram2d(one_ini[:,1], one_ini[:,2], bins=(hone_ene,hone_cir))
        plt.imshow(heatone_ini.T, origin='lower',aspect=one_emax,cmap='jet',extent=(0,one_emax,0,one_cmax),vmax=1,vmin=0)
        # plt.colorbar(label='Number of Bound Particles')
        plt.title(f'Initially Bound Particles from Component 1')
        plt.xlabel('Energy')
        plt.ylabel('L/Lmax')
        if save_plot == True: plt.savefig(f'./orbit_one_000',dpi=400,transparent=False)
        # plt.show()
        plt.close('all')

        plt.clf()
        heattwo_ini, xtwo_ini, ytwo_ini = np.histogram2d(two_ini[:,1], two_ini[:,2], bins=(htwo_ene,htwo_cir))
        plt.imshow(heattwo_ini.T, origin='lower',aspect=two_emax,cmap='jet',extent=(0,two_emax,0,two_cmax),vmax=1,vmin=0)
        # plt.colorbar(label='Number of Bound Particles')
        plt.title(f'Initially Bound Particles from Component 2')
        plt.xlabel('Energy')
        plt.ylabel('L/Lmax')
        if save_plot == True: plt.savefig(f'./orbit_two_000',dpi=400,transparent=False)
        # plt.show()
        plt.close('all')