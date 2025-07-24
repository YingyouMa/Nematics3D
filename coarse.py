import numpy as np
import time
import h5py
import gzip
import os
from pathlib import Path

from Nematics3D.lammps import read_lammps
from Nematics3D.field import diagonalizeQ

# TODO: support non-cubic box
# ! Remember to reshape qtensor before save in the future version

# Constants of filaments
DENSITY 	= 0.7
NATOMS 		= 50



# ------------------------------------------------------------------------
# Read the coordinates of each monomer and calculate some basic parameters
# ------------------------------------------------------------------------

def read_pos(frame, path, prefix, suffix):

    data, bounds = read_lammps(path+prefix+str(frame)+suffix)
    data.sort_values(by='id', inplace=True)
    NUM_ATOMS = len(data)
    num_polys = np.max(data['mol'])
    length = NUM_ATOMS // num_polys  # polymer length
    xlo, xhi = bounds['x']
    ylo, yhi = bounds['y']
    zlo, zhi = bounds['z']
    LX = xhi - xlo
    LY = yhi - ylo
    LZ = zhi - zlo
    r = data[['xu', 'yu', 'zu']].values.copy()
    r -= np.array([xlo,ylo,zlo])
    r %= [LX, LY, LZ]

    return r, LX, LY, LZ, length, num_polys, NUM_ATOMS


# -----------------------------------------
# Derive the raw density and Q tensor field
# -----------------------------------------

def count_monomer(r, length, sdt, L, VOXEL, N):

    N = np.array([N]).reshape(-1)
    if len(N) == 1:
        NX, NY, NZ = N[0], N[0], N[0]
    elif len(N) == 3:
        NX, NY, NZ = N

    # Find each local orientation of bond
    # If the bond length is too big, the neighboring monomers are crossing the simulation box
    p = np.gradient(r.reshape([length, -1], order='F'), axis=0).reshape([-1, 3], order='F')
    I = p > sdt; p[I] -= L
    I = p < -sdt; p[I] += L
    I = (p < sdt) * (p > sdt/2); p[I] -= L/2
    I = (p > -sdt) * (p < -sdt/2); p[I] += L/2
    p = p /np.linalg.norm(p, axis=1)[:,None]
    del I
    
    # Derive the density field
    loc = np.round(r / VOXEL).astype(int)
    loc[:,0] %= NX; loc[:,1] %= NY;  loc[:,2] %= NZ;
    loc = tuple(loc.T) 
    cnt = np.zeros((NX,NY,NZ), dtype=int)
    np.add.at(cnt, loc, 1)
    den = cnt / np.product(VOXEL)
    
    # Derive the Q tensor field
    M = np.einsum('ij,ik->ijk', p, p)
    qtensor = np.zeros((3,3,NX,NY,NZ))
    np.add.at(qtensor.transpose([2,3,4,0,1]), loc, M)
    qtensor /= np.product(VOXEL)

    return den, qtensor


# ----------------------------------------------------
# Gaussian filter working on Fourier transformed field
# ----------------------------------------------------

def kernal_fft(fp, sig, L):
    '''
    fp: Fourier transformed field. Shape: (..., N, N, N)
    '''

    N = fp.shape[-3]
    F = fp[...] *N**3
    
    kx = np.fft.fftfreq(N, L /N)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftfreq(N, L /N)
    ky = np.fft.fftshift(ky)
    kz = np.fft.rfftfreq(N, L /N)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    kernel = np.exp(-(Kx**2+Ky**2+Kz**2) *sig**2 *(2 *np.pi)**2 /2)
    
    F = np.einsum('...jkl, jkl->...jkl', F, kernel)
    
    F = F[..., :int(N/2)+1]
    return F


# --------------------------------------------------------------
# Coarse-grain the field by truncate at some maximum wave number
# --------------------------------------------------------------

def truncate_rfft_coefficients(F, new_NX, new_NY, new_NZ):
    if (new_NX%2 or new_NY%2 or new_NZ%2):
        raise ValueError("NX, NY and NZ must be even.")
    NX = F.shape[-3]
    NY = F.shape[-2]
    NZ = (F.shape[-1] - 1) *2
    if (new_NX > NX or new_NY > NY or new_NZ > NZ):
        raise ValueError("New array dimensions larger or equal to input dimensions.")
    if (new_NX == NX and new_NX == NY and new_NZ == NZ):
        return F
    mid_x = NX //2
    mid_y = NY //2
    s = (...,
         slice(mid_x-new_NX//2, mid_x+new_NX//2),
         slice(mid_y-new_NY//2, mid_y+new_NY//2),
         slice(0, new_NZ//2+1), 
         )
    tmp = np.fft.fftshift(F, axes=(-2,-3))
    tmp = tmp[s]
    # tmp = np.fft.ifftshift(tmp, axes=0) /NX /NY
    return tmp /NX /NY /NZ


# -----------------------------------
# IFFT the density and Q tensor field
# -----------------------------------

def IFFT_nematics(Fq, Fd=0, N_out=0, if_make_traceless=True):
    
    N_trunc = Fq.shape[-3]

    if N_out == 0:
        N_out = N_trunc
        
    xpad = int((N_out-N_trunc)/2)
    ratio = (N_trunc + 2*xpad) / N_trunc

    Fq = np.pad(Fq, ((0,0), (xpad, xpad), (xpad, xpad), (0, xpad)))
    Fq = np.fft.fftshift(Fq, axes=(-3,-2))
    qtensor = np.fft.irfftn(Fq, axes=(-3,-2,-1)) * ratio**3
    if if_make_traceless==True:
        qtensor[0] -= 1/3
        qtensor[3] -= 1/3

    if isinstance(Fd, int) == False:
        Fd = np.pad(Fd, ((xpad, xpad), (xpad, xpad), (0, xpad)))
        Fd = np.fft.fftshift(Fd, axes=(-3,-2))
        den = np.fft.irfftn(Fd) * ratio**3
        qtensor /= den[None,...]
    else:
        den = None

    return qtensor, den


# ---------------------------------------------------------------------------
# The main function to coarse-graining one frame of particle-based simulation 
# ---------------------------------------------------------------------------

def coarse_one_frame(
                    address, save_path,
                    stiffness, activity, name, frame, prefix='', suffix='.data', 
                    N_raw=300, N_trunc=128, sdtn=0.9,
                    if_IFFT=True, sig=2, N_out=128,
                    diag_path=0
                    ):

    start = time.time()

    path = address + '/dump/'

    N_raw = np.array([N_raw]).reshape(-1)
    if len(N_raw) == 1:
        NX, NY, NZ = N_raw[0], N_raw[0], N_raw[0]
    elif len(N_raw) == 3:
        NX, NY, NZ = N_raw

    # Read the coordinates
    r, LX, LY, LZ, length, num_polys, NUM_ATOMS = read_pos(frame, path, prefix, suffix)

    VOXEL = np.array([LX, LY, LZ]) / [NX, NY, NZ]
    sdt = sdtn * LX

    # Derive the raw density and Q tensor field
    den, qtensor = count_monomer(r, length, sdt, LX, VOXEL, [NX, NY, NZ])

    # Fourier transform the density field and truncate it at maximum wave number
    F_density = np.zeros(shape=(NX,NY,NZ//2+1), dtype='complex128')
    F_density = np.fft.rfftn(den)
    F_density = truncate_rfft_coefficients(F_density, N_trunc, N_trunc, N_trunc)
    del den
    
    # Fourier transform the Q tensor field and truncate it at maximum wave number
    F_qtensor = np.zeros(shape=(5,NX,NY,NZ//2+1), dtype='complex128')
    F_qtensor[0] = np.fft.rfftn(qtensor[0,0])
    F_qtensor[1] = np.fft.rfftn(qtensor[0,1])
    F_qtensor[2] = np.fft.rfftn(qtensor[0,2])
    F_qtensor[3] = np.fft.rfftn(qtensor[1,1])
    F_qtensor[4] = np.fft.rfftn(qtensor[1,2])
    F_qtensor = truncate_rfft_coefficients(F_qtensor, N_trunc, N_trunc, N_trunc)
    del qtensor

    # Store the FFT results
    Path(save_path+'/FFT').mkdir(exist_ok=True, parents=True)
    with h5py.File(save_path+'/FFT/'+str(frame)+'.h5py', 'w') as f:
    
        f.create_dataset('qtensor',  dtype='complex128', data=F_qtensor)
        f.create_dataset('density',  dtype='complex128', data=F_density)
    
        params = {
                    "grid_N": (NX, NY, NZ), "FFT_truncate": (N_trunc, N_trunc, N_trunc), \
                    "LX": LX, "LY": LY, "LZ": LZ, \
                    "num_polys": num_polys, "num_atoms": NUM_ATOMS,  \
                    "data_path": path, "stiffness": stiffness, "activity": activity, "name": name, "frame": frame
                    }
        f.create_dataset('params', data=str(params))

    if if_IFFT == True:

        Path(save_path+f'/result_{N_out}').mkdir(exist_ok=True)

        Fd = kernal_fft(F_density, sig, LX)
        Fq = kernal_fft(F_qtensor, sig, LX)

        qtensor, den = IFFT_nematics(Fq, Fd=Fd, N_out=N_out)
        qtensor = qtensor.transpose((1,2,3,0))
        

        with h5py.File(save_path+f'/result_{N_out}/'+str(frame)+'.h5py', 'w') as fw:
            fw.create_dataset('density', data=den)
            fw.create_dataset('qtensor', data=qtensor)
            fw.create_dataset('sigma', data=sig)

        if diag_path != 0:

            Path( diag_path + f"/{N_out}/" ).mkdir(exist_ok = True, parents=True)

            S, n = diagonalizeQ(qtensor)

            np.save( diag_path + f"/{N_out}/S_{frame}.npy", S )
            np.save( diag_path + f"/{N_out}/n_{frame}.npy", n )
    
    # Zip the analyzed file
    unzip_file  = path + prefix + str(frame) + suffix
    zip_file    = path + prefix + str(frame) + suffix + '.gz'
    with open(unzip_file, 'rb') as f_in:
        content = f_in.read()
    f = gzip.open( zip_file, 'wb')
    f.write(content)
    f.close()
    if os.path.isfile(zip_file):
        os.remove(unzip_file)

    print(frame, round(time.time()-start,2), 's')   
    