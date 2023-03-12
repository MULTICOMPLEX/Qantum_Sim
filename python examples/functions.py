import numpy as np
from constants import *
import pyfftw
import multiprocessing

def fft_frequencies(N, dx, hbar):
    p1 = np.fft.fftfreq(N, d = dx) * hbar  * 2*np.pi
    p2 = np.fft.fftfreq(N, d = dx) * hbar  * 2*np.pi
    p1, p2 = np.meshgrid(p1, p2)
    p2 = (p1**2 + p2**2)
    return p2

def norm(phi, dx):
    norm = np.linalg.norm(phi) * dx
    return (phi * np.sqrt(dx)) / norm

def norm2(phi, dx):
    return phi/np.sqrt(np.sum(np.square(np.abs(phi)) * dx))

#P = sum_i |psi_i><psi_i|
#method for projecting a vector onto a given subspace.
#orthogonal projection 
def apply_projection(tmp, psi_list, dx):
    for psi in psi_list:
        tmp -= np.vdot(psi,tmp) * psi * dx   
    return tmp
    
def apply_projection2(tmp, psi_list, dx):
    for psi in psi_list:
        tmp -= np.sum(tmp*psi.conj()) * psi * dx 
    return tmp

def differentiate_twice(f, p2):
    f = np.fft.ifftn(-p2*np.fft.fftn(f))
    return f

'''
def differentiate_once(f, p):
    f = np.fft.ifft(1j*p*np.fft.fft(f))
    return f

def integrate_twice(f, p2):
    F = np.fft.fftn(f)
    F[1:] /= p2[1:]
    F = np.fft.ifftn(F)
    F -= F[0]
    return F
    
def integrate_once(f, p):
    # Compute the FFT of the function
    F = np.fft.fftn(f)
    # Divide each coefficient by the constant for the corresponding frequency
    F[1:] /= p[1:]
    return np.fft.ifftn(F)
'''
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()

def Split_Step_FFTW(Ψ, phi, dx, store_steps, Nt_per_store_step, Ur, Uk, ite):
    for i in range(store_steps):
        tmp = Ψ[i]
        for _ in range(Nt_per_store_step):
            c = pyfftw.interfaces.numpy_fft.fftn(Ur*tmp)
            tmp = Ur * pyfftw.interfaces.numpy_fft.ifftn(Uk*c)
            if(ite):
              tmp = norm(apply_projection(tmp, phi, dx), dx)
        Ψ[i+1] = tmp
    return

def Split_Step_NP(Ψ, phi, dx, store_steps, Nt_per_store_step, Ur, Uk, ite):
    for i in range(store_steps):
        tmp = Ψ[i]
        for _ in range(Nt_per_store_step):
            c = np.fft.fftn(Ur*tmp)
            tmp = Ur * np.fft.ifftn(Uk*c)
            if(ite):
              tmp = norm(apply_projection(tmp, phi, dx), dx)
        Ψ[i+1] = tmp
    return
