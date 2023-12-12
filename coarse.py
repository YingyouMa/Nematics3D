import numpy as np

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

# -----------------------------------
# IFFT the density and Q tensor field
# -----------------------------------


def IFFT(Fd, Fq, N_out=0):
    
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
    
