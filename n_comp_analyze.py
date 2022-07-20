import struct,os
# from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import integrate
from general_functions import *
import glob

folder = 'one_component'
# folder = 'two_component'
# folder = 'S3'

num = 0 # file number that you want to analyze
overwrite = True # overwrites old data files
save_plot = True
if not(os.path.exists(f'./{folder}/analytics/energy')): os.makedirs(f'./{folder}/analytics/energy')
if not(os.path.exists(f'./{folder}/analytics/circularity')): os.makedirs(f'./{folder}/analytics/circularity')

sizes = np.loadtxt(f"./{folder}/analytics/component_sizes.txt")
sizes = np.array(sizes[1:]) - np.array(sizes[:-1])
comp = sizes.tolist()
comp.insert(0,0)

# # =======================================================================
# # Open Data Files
# # =======================================================================
#
# # sets up string segment and opens file
# numtxt = str(num)
# while len(numtxt) < 3: numtxt = '0' + numtxt
#
# filename = f'./{folder}/snapshots/snapshot_{numtxt}'
# header, Numfiles = loadgadget_header(filename)
# header, data = loadgadget(filename)
# snap_data = data.copy()
# m = header['mass']
# G = 1
# rho_0 = (data[:,0].size * m)/(4 * np.pi * (1/(1+10) -1 +np.log(11))) # fix this at some point
# scale_radius = 1.0
#
# data_bound = np.loadtxt(f"./{folder}/analytics/bound/bound_{numtxt}.txt")
# data_bound = np.loadtxt(f"./{folder}/analytics/bound/bound_{numtxt}.txt")
#
# snap_bound = snap_data[np.isin(snap_data[:,0],data_bound)]
#
# # uses last snapshot's bound particle data to filter particles from currently opened file
# data_bound = data[np.isin(data[:,0],data_bound)]
# data_unbnd = data[np.isin(data[:,0],data_bound,invert=True)]
# # plt.scatter(data_unbound[:,1],data_unbound[:,2],marker='.')
# # plt.scatter(data_bound[:,1],data_bound[:,2],marker='.')
# # plt.ylim(-300,300)
# # plt.xlim(-300,300)
# # plt.show()
# # plt.close('all')
#
# # =======================================================================
# # Volumes and Masses
# # =======================================================================
#
# log_frame = np.loadtxt(f"./{folder}/analytics/frame.txt")
#
# # find frame and shift points
# data_bound[:,1:] -= log_frame[num,:]
# data_unbnd[:,1:] -= log_frame[num,:]
# bound_dist = np.sqrt(data_bound[:,1]**2+data_bound[:,2]**2+data_bound[:,3]**2)
# unbnd_dist = np.sqrt(data_unbnd[:,1]**2+data_unbnd[:,2]**2+data_unbnd[:,3]**2)
#
# # groups particles into bins based on distance from center
# edges = np.logspace(-1,1,10)
#
# # graphs visual for bins
# # fig, ax = plt.subplots()
# # ax.scatter(data_bound[:,1],data_bound[:,2],marker='.')
# # for i in edges:
# #     ax.add_patch(plt.Circle((0,0),i,fill=False))
# # plt.show()
# # plt.close('all')
#
# particles, bins = np.histogram(bound_dist,bins=edges)
# particle_mass = particles * m
#
# volumes = np.zeros(len(particles))
# distance = (bins[:-1]+bins[1:])/2
#
# # add up all the particles between each bin value
# for i in range(len(particles)):
#     value, _ = integrate.quad(vol_sphere,bins[i],bins[i+1])
#     volumes[i] = value
#
# plt.xticks(bins)
# plt.hist(bound_dist,bins=bins)
# if save_plot == True: plt.savefig(f'./{folder}/analytics/density_{numtxt}',dpi=400,transparent=False)
# # plt.show()
# plt.close('all')

# # ========================================================================
# # Calculate Momentum and Energy
# # ========================================================================
# # Opens analytics (*_energy and *_circularity files if they exist. If not,
# # generates files and saves them in ./analytics/ for future usage
#
# if overwrite & os.path.exists(f'./analytics/energy/snap_energy_{numtxt}'):
#     print('Overwriting analytics files...')
#     os.remove(f'./{folder}/analytics/energy/snap_energy_{numtxt}')
#     os.remove(f'./{folder}/analytics/energy/snap_energy_{numtxt}')
#
# if not(os.path.exists(f'./{folder}/analytics/energy/snap_energy_{numtxt}')):
#     print('Creating analytics...')
#     momentum = np.cross(data_bound[:,1:4],data_bound[:,4:7])*m
#     data_mom = np.sqrt(momentum[:,0]**2+momentum[:,1]**2+momentum[:,2]**2)
#
#     data_pos = np.array(np.sqrt(data_bound[:,1]**2+data_bound[:,2]**2+data_bound[:,3]**2))
#     data_vel = np.array(np.sqrt(data_bound[:,4]**2+data_bound[:,5]**2+data_bound[:,6]**2))
#     data_ene = -(calculate_potential_spherical(data_pos,G,m) + 1/2*data_vel**2)
#
#     data_cir = np.zeros(len(data_bound[:,0]))
#     for i in range(len(data_cir)):
#         data_cir[i] = NFW_circ(data_bound[i,1:4],data_bound[i,4:7],rho_0,1.0,G)
#
#     np.savetxt(f"./{folder}/analytics/energy/snap_energy_{numtxt}.txt", data_ene, fmt="%s")
#     np.savetxt(f"./{folder}/analytics/circularity/snap_circularity_{numtxt}.txt", data_cir, fmt="%s")
#
# else:
#     data_ene = np.loadtxt(f"./{folder}/analytics/energy/snap_energy_{numtxt}.txt")
#     data_cir = np.loadtxt(f"./{folder}/circularity/energy/snap_circularity_{numtxt}.txt")
#
# edges_yaxis = np.linspace(0,np.max(data_cir),25)
# edges_xaxis = np.linspace(0,np.max(data_ene),25)
#
# heatmap, xedges, yedges = np.histogram2d(data_ene, data_cir, bins=(edges_xaxis,edges_yaxis))
#
# plt.clf()
# # plt.hist2d(np.log10(parts_ene),np.log10(circ),bins=bin_edges)
# plt.imshow(np.log10(heatmap.T), origin='lower',aspect='auto',extent=(edges_xaxis[0],edges_xaxis[-1],edges_yaxis[0],edges_yaxis[-1]))
# plt.title(f'Snapshot_{numtxt} - Energy vs Circularity')
# plt.xlabel('Energy')
# plt.ylabel('L/Lmax')
# if save_plot == True: plt.savefig(f'./{folder}/analytics/energy_{numtxt}',dpi=400,transparent=False)
# # plt.show()
# plt.close('all')

# =======================================================================
# Energy vs Circularity Comparisons
# =======================================================================

filename = f'./{folder}/snapshots/snapshot_000'
header, Numfiles = loadgadget_header(filename)
header, data = loadgadget(filename)

snap_count = len(glob.glob(f'./{folder}/snapshots/snapshot_*'))
interval = 1

# ene_max = 0.04
# cir_max = 1
ene_max = 0.4
cir_max = 1

particle_count = data.shape[0]
G = 1

bins_ene = np.linspace(0,ene_max,100)
bins_cir = np.linspace(0,cir_max,100)

for i in range(snap_count):
    if i != 0:
        if i % interval == 0:
            itxt = str(i)
            while len(itxt) < 3: itxt = '0' + itxt
            orbit = str(int(i/interval))

            bound_txt = np.loadtxt(f"./{folder}/analytics/bound/bound_{itxt}.txt")
            bound_orb = np.intersect1d(initial[:,0],bound_txt)
            orbital = initial[np.isin(initial[:,0],bound_orb)]

            heatmap_orb, x_orb, y_orb = np.histogram2d(orbital[:,1], orbital[:,2], bins=(bins_ene,bins_cir))
            heatmap_com = heatmap_orb/heatmap_ini
            heatmap_com[np.isnan(heatmap_com)] = 0
            plt.imshow(heatmap_com.T, origin='lower',aspect=ene_max,cmap='jet',vmin=0,vmax=1,extent=(0,ene_max,0,cir_max))
            plt.colorbar(label='Fraction of Initially Bound Particles')
            plt.title(f'Remainder of Bound Particles - Snapshot_{itxt}')
            plt.xlabel('Energy')
            plt.ylabel('L/Lmax')
            plt.savefig(f'./{folder}/animations/energyspace/energyspace_{itxt}',dpi=400,transparent=False)
            plt.close('all')

            orbital_one = orbital[orbital[:,0] > comp[1]]
            orbital_two = orbital[orbital[:,0] <= comp[1]]

            heatone_orb, x_orb, y_orb = np.histogram2d(orbital_one[:,1], orbital_one[:,2], bins=(bins_ene,bins_cir))
            heatone_com = heatone_orb/heatone_ini
            heatone_com[np.isnan(heatone_com)] = 0
            plt.imshow(heatone_com.T, origin='lower',aspect=ene_max,cmap='jet',vmin=0,vmax=1,extent=(0,ene_max,0,cir_max))
            plt.colorbar(label='Fraction of Initially Bound Particles')
            plt.title(f'Component 1: Remainder of Bound Particles - Snapshot_{itxt}')
            plt.xlabel('Energy')
            plt.ylabel('L/Lmax')
            plt.savefig(f'./{folder}/animations/energyspace/energyspace_one_{itxt}',dpi=400,transparent=False)
            plt.close('all')

            heattwo_orb, x_orb, y_orb = np.histogram2d(orbital_two[:,1], orbital_two[:,2], bins=(bins_ene,bins_cir))
            heattwo_com = heattwo_orb/heattwo_ini
            heattwo_com[np.isnan(heattwo_com)] = 0
            plt.imshow(heattwo_com.T, origin='lower',aspect=ene_max,cmap='jet',vmin=0,vmax=1,extent=(0,ene_max,0,cir_max))
            plt.colorbar(label='Fraction of Initially Bound Particles')
            plt.title(f'Component 2: Remainder of Bound Particles - Snapshot_{itxt}')
            plt.xlabel('Energy')
            plt.ylabel('L/Lmax')
            plt.savefig(f'./{folder}/animations/energyspace/energyspace_two_{itxt}',dpi=400,transparent=False)
            plt.close('all')

    else:
        bound = np.loadtxt(f"./{folder}/analytics/bound/bound_000.txt")
        energy = np.loadtxt(f"./{folder}/analytics/energy/energy_000.txt")
        circularity = np.loadtxt(f"./{folder}/analytics/circularity/circularity_000.txt")
        initial = np.column_stack((bound,energy,circularity))

        heatmap_ini, xedges, yedges = np.histogram2d(initial[:,1], initial[:,2], bins=(bins_ene,bins_cir))
        plt.imshow(heatmap_ini.T, origin='lower',aspect=ene_max,cmap='jet',extent=(0,ene_max,0,cir_max),vmin=0,vmax=1)
        # plt.colorbar(label='Number of Bound Particles')
        plt.title(f'Initially Bound Particles')
        plt.xlabel('Energy')
        plt.ylabel('L/Lmax')
        plt.savefig(f'./{folder}/analytics/initial_energyspace',dpi=400,transparent=False)
        # plt.show()
        plt.close('all')

        initial_one = initial[initial[:,0] > comp[1]]
        initial_two = initial[initial[:,0] <= comp[1]]

        heatone_ini, xedges, yedges = np.histogram2d(initial_one[:,1], initial_one[:,2], bins=(bins_ene,bins_cir))
        plt.imshow(heatone_ini.T, origin='lower',aspect=ene_max,cmap='jet',extent=(0,ene_max,0,cir_max),vmin=0,vmax=1)
        plt.title(f'Component 1: Initially Bound Particles')
        plt.xlabel('Energy')
        plt.ylabel('L/Lmax')
        plt.savefig(f'./{folder}/analytics/initial_energyspace_one',dpi=400,transparent=False)
        plt.close('all')

        heattwo_ini, xedges, yedges = np.histogram2d(initial_two[:,1], initial_two[:,2], bins=(bins_ene,bins_cir))
        plt.imshow(heattwo_ini.T, origin='lower',aspect=ene_max,cmap='jet',extent=(0,ene_max,0,cir_max),vmin=0,vmax=1)
        plt.title(f'Component 2: Initially Bound Particles')
        plt.xlabel('Energy')
        plt.ylabel('L/Lmax')
        plt.savefig(f'./{folder}/analytics/initial_energyspace_two',dpi=400,transparent=False)
        plt.close('all')

os.system(f"ffmpeg -framerate 4 -y -i ./{folder}/animations/energyspace/energyspace_%03d.png ./{folder}/animations/energyspace_stripping.mp4")
os.system(f"ffmpeg -framerate 4 -y -i ./{folder}/animations/energyspace/energyspace_one_%03d.png ./{folder}/animations/energyspace_stripping_one.mp4")
os.system(f"ffmpeg -framerate 4 -y -i ./{folder}/animations/energyspace/energyspace_two_%03d.png ./{folder}/animations/energyspace_stripping_two.mp4")
