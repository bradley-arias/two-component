import struct,os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import integrate
from scipy import optimize

def loadgadget_header(filename):
    '''
    Code to read in the header of a Gadget File
    ---------
    Inputs
        filename: base of filename (if split into multiple files, leave off '.x')
    ---------
    Outputs
        header: dictionary with snapshot info
        Numfiles: number of files for the snapshot
    '''

    #Open file
    if os.path.exists(filename):
        f = open(filename,'rb')
    else:
        f = open(filename+'.0','rb')

    #Read header
    myheader = f.read(256+4)
    N_arr = np.array(struct.unpack('i'*6,myheader[4:28]))
    m_arr =  np.array(struct.unpack('d'*6,myheader[28:76]))
    time = struct.unpack('d',myheader[76:84])[0]
    redshift = struct.unpack('d',myheader[84:92])[0]
    N_all = np.array(struct.unpack('i'*6,myheader[100:124]))
    Numfiles = struct.unpack('i',myheader[128:132])[0]
    Boxsize = struct.unpack('d',myheader[132:140])[0]
    Omega0 = struct.unpack('d',myheader[140:148])[0]
    OmegaLam = struct.unpack('d',myheader[148:156])[0]
    H0 = struct.unpack('d',myheader[156:164])[0]
    f.close()

    #Create header output
    header = {
      "N": N_all[1],
      "mass": m_arr[3],
      "time": time
    }

    return header, Numfiles

def loadgadget(filename,longIDs=False):
    '''
    Load gadget file
    ---------
    Inputs
        filename: base of filename (if split into multiple files, leave off '.x')
        longIDs: if IDs are stored as 64 bit integers, have to turn this on
    ---------
    Outputs
        header: dictionary with snapshot info
        data: Nx7 array with ID, x, y, z, vx, vy, vz
    '''

    header, Numfiles = loadgadget_header(filename)
    data = np.array([])

    for i in range(Numfiles):

        #Open file
        if Numfiles>1:
            myfilename=filename+'.'+str(i)
        else:
            myfilename=filename
        f = open(myfilename,'rb')

        #Read header
        myheader = f.read(256+4)
        N_arr = np.array(struct.unpack('i'*6,myheader[4:28]))
        f.read(8)

        #Set up mydata
        mydata = np.zeros([N_arr[1],7])

        #Read positions
        temp = f.read(4*3*N_arr[1])
        temp = np.ndarray((1, 3*N_arr[1]), 'f', temp)[0]
        mydata[:,1:4] = np.reshape(temp,(N_arr[1],3))
        f.read(8)

        #Read velocities
        temp = f.read(4*3*N_arr[1])
        temp = np.ndarray((1, 3*N_arr[1]), 'f', temp)[0]
        mydata[:,4:7] = np.reshape(temp,(N_arr[1],3))
        f.read(8)

        #Read IDs
        if longIDs:
            temp = f.read(8*N_arr[1])
            temp = np.ndarray((1, N_arr[1]), 'l', temp)[0]
        else:
            temp = f.read(4*N_arr[1])
            temp = np.ndarray((1, N_arr[1]), 'i', temp)[0]
        mydata[:,0] = abs(temp)
        f.read(8)

        if header['mass']==0:
            #Read Masses
            temp = f.read(4*N_arr[1])
            temp = np.ndarray((1,N_arr[1]), 'f', temp)[0]
            if np.all(temp  == temp[0]):
                mass = temp[0]
                header['mass']=mass
            else:
                mass = temp

        f.close()

        if data.shape[0]==0:
            data = mydata.copy()
        else:
            data = np.vstack([data,mydata])


    return header, data

# def clustercentre(pos, R, N_min,alpha,reps_max=1e3):
#     '''
#     Finds densest point
#     ---------
#     Inputs
#         pos: numpy array. position, (x,y,z) for each particle. Size: Nx3
#         R: float. size of sphere. Choose this to be about the size of the subhalo.
#         N_min: Minimum number of particles in sphere before exiting the loop
#         alpha: how much to reduce the radius by on each iteration
#         reps_max: maximum number of iterations.
#     ---------
#     Outputs
#         center: numpy array, with the center [x,y,z]
#     '''


#     N = pos.shape[0] #number of points
#     p = pos.copy() #initialize
#     center = np.zeros([1,3]) #initialize

#     reps = 0 #counter
#     while (N>N_min):
#         center += [np.mean(p[:,0]), np.mean(p[:,1]), np.mean(p[:,2])] #find new COM
#         R = alpha*R #reduce radius
#         p = pos-center #shift points to new COM
#         r = np.sqrt(p[:,0]*p[:,0] + p[:,1]*p[:,1] + p[:,2]*p[:,2]) #find r of points
#         p = p[r<R] #only consider points within radius
#         N  = p.shape[0]
#         reps+=1

#         if reps==reps_max:
#             print("clustercentre didn't converge")
#             break

#     return center[0]

def clustercentre(pos, N_min,alpha,R=-1,reps_max=1e3):
    '''Finds densest point
    ---------
    Inputs
        pos: numpy array. position, (x,y,z) for each particle. Size: Nx3
        N_min: Minimum number of particles in sphere before exiting the loop
        alpha: how much to reduce the radius by on each iteration
        reps_max: maximum number of iterations.
    ---------
    Outputs
        center: numpy array, with the center [x,y,z]'''

    #initialize
    N = pos.shape[0] #number of points
    p = pos.copy()
    center = np.zeros([1,3])
    reps = 0 #counter
    while (N>N_min):
        center += [np.mean(p[:,0]), np.mean(p[:,1]), np.mean(p[:,2])] #find new COM
        p = pos-center #shift points to new COM
        r = np.sqrt(p[:,0]*p[:,0] + p[:,1]*p[:,1] + p[:,2]*p[:,2]) #find r of points
        if R<0:
            R = max(r)
        else:
            p = p[r<R] #only consider points within radius
        N  = p.shape[0]
        R = alpha*R #reduce radius
        reps+=1
        if reps==reps_max:
            print("clustercentre didn’t converge")
            break
    return center[0]

def calculate_potential_spherical(r,G,m):
    '''
    Calculates the potential of an N-body system, assuming its's spherical
    ---------
    Inputs
        r: numpy array of length N... radial distance of each particle
        G: gravitational constant
        m: mass of each particle
    ---------
    Outputs
        P: potential at position of each particle (numpy array, length N)
    '''

    r[r==0]+=1e-8 #make sure no zeros.

    #Inside Potential
    R_sort, R_ind = np.unique(r, return_inverse=True) #sorted list without repeats, index
    par_int = (np.cumsum(np.concatenate(([0], np.bincount(R_ind)))))[R_ind] #number of interior particles
    P = par_int/r
    #Outside Potential
    counter = Counter(r) # find repeated values
    vals = counter.values() #number of repeats of value in keys
    keys = counter.keys()
    vals = np.array(list(vals), dtype=float) #Convert to arrays
    keys = np.array(list(keys), dtype=float)
    inds = keys.argsort() # increasing order of keys
    reps = vals[inds] #get number of repeats in increasing order
    R_out = reps[::-1] * 1/R_sort[::-1]
    R_out = np.cumsum(R_out) - R_out #sum 1/R for all exterior particles
    R_out = R_out[::-1] #flip direction
    P += R_out[R_ind]
    P = -G*m*P
    return P

def energies_spherical(data,G,m,P=0.0,P0=0.0):
    '''
    Calculate the energy of each particle: assumes a spherical potential if P not specified
    ---------
    Inputs
    data: numpy array, Nx7... contains particle ID,x,y,z,vx,vy,vz
        G: gravitational constant
        m: mass of each particle
        P: can pass the potential of each particle if it is precalculated...otherwise it will calculate the potential assuming sphericity
        P0: can shift the energy by a constant P0
    ---------
    Outputs
        E: Energy of each each particle. Energy is defined as in Binney and Tremaine as E=-(P+K), negative energies mean the particle is unbound
    '''

    ID,x,y,z,vx,vy,vz = data.T
    r = np.linalg.norm([x,y,z],axis=0)
    v = np.linalg.norm([vx,vy,vz],axis=0)

    if isinstance(P,float) or isinstance(P,int):
        P=calculate_potential_spherical(r,G,m)
    E=(-P-P0) - v*v/2.0

    return E

def remove_unbound(data,G,m,r_max=np.inf,P0=0.0):
    '''
    Removes particles with binding energy less than P0. Assumes spherical potential
    ---------
    Inputs
        data: numpy array, Nx7... contains particle ID,x,y,z,vx,vy,vz
        G: gravitational constant
        m: mass of each particle
        r_max: only include particles within this radius (default infinity)
        P0:  minimum energy of bound particles (default zero)
    ---------
    Outputs
        E: data_bound: N_boundx7... numpy array containing unbound particles
    '''
    data_bound = data.copy()
    nnew = data.shape[0]
    nold = nnew + 1

    while nold>nnew:

        #Remove particles outside of r_max
        ID,x,y,z,vx,vy,vz = data_bound.T
        r = np.linalg.norm([x,y,z],axis=0)
        v = np.linalg.norm([vx,vy,vz],axis=0)
        data_bound = data_bound[r<r_max]

        #Remove particles with KE>PE
        E = energies_spherical(data_bound,G,m,P0)
        data_bound = data_bound[E>0]

        #Get new number of particles
        nold = int(nnew)
        nnew = data_bound.shape[0]

    return data_bound

def findvelocities(location):
    file_list = os.listdir(location)
    for item in range(len(file_list)):
        if item > 9:
            num = '0' + str(item)
        else:
            num = '00' + str(item)
        filename = f'{location}snapshot_{num}'
        header, Numfiles = loadgadget_header(filename)
        header, data = loadgadget(filename)
        ID,x,y,z,vx,vy,vz = data.T
        av = np.sqrt((vx**2 + vy**2 + vz**2))

        if num != '000':
            if avmax < np.max(av):
                avmax = np.max(av)
            if avmin > np.min(av):
                avmin = np.min(av)
        else:
            avmax = np.max(av)
            avmin = np.min(av)
    return avmin, avmax

def func_NFW(radius,rho_0,scale_radius):
    return (rho_0*scale_radius**3)/(radius*(radius+scale_radius)**2)

def func_relaxtime(number,size,mass,gravity=1):
    return 0.1*(np.sqrt(number)/np.log(number))*np.sqrt(size**3/(gravity*mass))

def func_relaxinv(number,time,mass,gravity=1):
    return np.cbrt((time*np.log(number))/((0.1*np.sqrt(number))**2*gravity*mass))

def func_evaptime(time):
    return 136*time

def vol_sphere(radius):
    return 4*np.pi*radius**2

def NFW_circ(r,v,p0,r_s,G):
    #########################################################
    # Finds circularity parameter for object in NFW potential
    # given a radius and velocity somewhere on the obit.
    # Circularity is L/Lc, Lc is angular momentum of circular
    # orbit with same energy
    #########################################################
    rc = NFW_rc(r,v,p0,r_s,G) #radius of circular orbit with same energy
    temp = np.cross(r,v)
    # L = np.sqrt(temp[:,0]**2+temp[:,1]**2+temp[:,2]**2)
    L = np.sqrt(temp[0]**2+temp[1]**2+temp[2]**2)
    Vc = np.sqrt(4.0*np.pi*G*p0*r_s*r_s*r_s*(1.0/rc*np.log(rc/r_s +1.0) - 1.0/(rc+r_s)))
    Lc = rc*Vc
    circ = L/Lc
    return circ

def NFW_rc(r_v,v_v,p0,r_s,G):
    #########################################################
    # Finds radius of circular orbit with same energy
    #in NFW potential
    #-----
    #r and v of orbit (at apocenter?)
    #p0, r_s: NFW parameters
    #G: gravitational constant
    #########################################################
    # r = np.sqrt(r_v[:,0]**2+r_v[:,1]**2+r_v[:,2]**2)
    # v = np.sqrt(v_v[:,0]**2+v_v[:,1]**2+r_v[:,2]**2)
    r = np.sqrt(r_v[0]**2+r_v[1]**2+r_v[2]**2)
    v = np.sqrt(v_v[0]**2+v_v[1]**2+r_v[2]**2)
    def myfunc(rc,r,v,p0,r_s,G):
        A = np.pi*G*p0*r_s*r_s*r_s
        Ec = -2.0*A*(1.0/rc*np.log(rc/r_s +1.0) + 1.0/(rc+r_s))
        E = 0.5*v*v - 4.0*A * np.log(r/r_s +1.0)/r #1/2 v^2 + phi(r)
        return E - Ec
    rc = optimize.root(myfunc,1.0*r,args=(r,v,p0,r_s,G)).x
    return rc
