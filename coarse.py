import numpy as np
import lammps

# ----------------------------------------------
# Load some basic information from one dump file
# ----------------------------------------------

def read_params(frame, path, suffix):
    data, bounds = lammps.read_lammps(path+str(frame)+suffix)
    NUM_ATOMS = len(data)
    num_polys = np.max(data['mol'])
    length = NUM_ATOMS // num_polys  # polymer length
    xlo, xhi = bounds['x']
    ylo, yhi = bounds['y']
    zlo, zhi = bounds['z']
    LX = xhi - xlo
    LY = yhi - ylo
    LZ = zhi - zlo
    return LX, LY, LZ, length, num_polys, NUM_ATOMS


# ------------------------------------
# Read the coordinates of each monomer
# ------------------------------------

def read_pos(frame, path, suffix):
    data, bounds = lammps.read_lammps(path+str(frame)+suffix)
    data.sort_values(by='id', inplace=True)
    xlo, xhi = bounds['x']
    ylo, yhi = bounds['y']
    zlo, zhi = bounds['z']
    LX = xhi - xlo
    LY = yhi - ylo
    LZ = zhi - zlo
    r = data[['xu', 'yu', 'zu']].values.copy()
    r -= np.array([xlo,ylo,zlo])
    r %= [LX, LY, LZ]
    return r


# -----------------------------------------
# Derive the raw density and Q tensor field
# -----------------------------------------

def count_monomer(r, length, sdt, L, VOXEL, N):

    if len(N) == 1:
        NX, NY, NZ = N, N, N
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

def IFFT_nematics(Fd, Fq, N_out=0):
    
    N_trunc = Fd.shape[-3]

    if N_out == 0:
        N_out = N_trunc
        
    xpad = int((N_out-N_trunc)/2)
    ratio = (N_trunc + 2*xpad) / N_trunc
     
    Fd = np.pad(Fd, ((xpad, xpad), (xpad, xpad), (0, xpad)))
    Fd = np.fft.fftshift(Fd, axes=(-3,-2))
    den = np.fft.irfftn(Fd) * ratio**3
    
    Fq = np.pad(Fq, ((0,0), (xpad, xpad), (xpad, xpad), (0, xpad)))
    Fq = np.fft.fftshift(Fq, axes=(-3,-2))
    qtensor = np.fft.irfftn(Fq, axes=(-3,-2,-1)) * ratio**3
    qtensor /= den[None,...]
    qtensor[0] -= 1/3
    qtensor[3] -= 1/3

    return den, qtensor
    
