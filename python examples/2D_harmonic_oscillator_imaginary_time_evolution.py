import numpy as np
import time
import progressbar
import pyfftw
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import animation
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb
import matplotlib.animation as animation


Å = 1.8897261246257702
femtoseconds = 4.134137333518212 * 10.
m_e = 1.0
hbar = 1.0
n = 512

NStates = 7

S = {
 "total time": 1 * femtoseconds,
 "store steps": 20,
 "σ": 1.0 * Å, #
 "v0": 64. * Å / femtoseconds, #initial_wavefunction momentum #64
 "initial wavefunction offset x": 0 * Å,
 "initial wavefunction offset y": 0 * Å,
 "N": n,
 "dt": 0.25,
 "extent": 15 * Å,#30
 "Number of States": NStates,
 "imaginary time evolution": True,
 "animation duration": 4, #seconds
 "save animation": True,
 "fps": 30,
 "path save": "./gifs/",
 "title": "2D harmonic oscillator"
}


x = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])
y = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])
dx = x[1] - x[0]
x, y = np.meshgrid(x,y)

def norm2(phi):
    return phi/np.sqrt(np.linalg.norm(phi)**2)

def norm(phi):
    return phi/np.sqrt(np.sum(np.square(np.abs(phi))*dx))

def apply_projection(tmp, psi):
    for p in psi:
        tmp -= np.vdot(p, tmp) * p * dx
    return tmp  

def apply_projection2(tmp, psi_list):
    for psi in psi_list:
        tmp -= np.sum(tmp*np.conj(psi)) * psi * dx
    return tmp

def ITE(phi, store_steps, Nt_per_store_step, Ur, Uk, proj, ite):
    for i in range(store_steps):
        tmp = Ψ[i]
        for _ in range(Nt_per_store_step):
            c = pyfftw.interfaces.numpy_fft.fftn(Ur*tmp)
            tmp = Ur * pyfftw.interfaces.numpy_fft.ifftn(Uk*c)
            if(proj):
             tmp = norm(apply_projection(tmp, phi))
            elif(ite):
             tmp = norm(tmp)
        Ψ[i+1] = tmp
    return

def ITEnp(phi, store_steps, Nt_per_store_step, Ur, Uk, proj, ite):
    for i in range(store_steps):
        tmp = Ψ[i]
        for _ in range(Nt_per_store_step):
            c = np.fft.fftn(Ur*tmp)
            tmp = Ur * np.fft.ifftn(Uk*c)
            if(proj):
             tmp = norm(apply_projection(tmp, phi))
            elif(ite):
             tmp = norm(tmp)
        Ψ[i+1] = tmp
    return


#potential energy operator
def V():
    kx = 0.02
    ky = 0.02
    return 0.5 * kx * x**2 + 0.5 * ky * y**2

#potential energy operator
def V2():
    m = m_e
    T = 0.5*femtoseconds
    w = 2*np.pi/T
    k = m* w**2
  
    print("oscillation_amplitude ", np.sqrt(m/k) * S["v0"]/Å, " amstrongs")

    return 0.5 * k * x**2    +    0.5 * k * y**2
    

#initial waveform
def 𝜓0():
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    p_x0 = m_e * S["v0"]
    σ = S["σ"]
    return np.exp( -1/(4* σ**2) * ((x-S["initial wavefunction offset x"])**2+
    (y-S["initial wavefunction offset y"])**2)) / np.sqrt(2*np.pi* σ**2)  *np.exp(p_x0*x*1j)
    

V = V() 
Vmin = np.amin(V)
Vmax = np.amax(V)


p1 = np.fft.fftfreq(S["N"], d = dx) * hbar  * 2*np.pi
p2 = np.fft.fftfreq(S["N"], d = dx) * hbar  * 2*np.pi
p1, p2 = np.meshgrid(p1, p2)
p2 = (p1**2 + p2**2)

    
dt_store = S["total time"] / S["store steps"]
Nt_per_store_step = int(np.round(dt_store / S["dt"]))
#time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
dt = dt_store/Nt_per_store_step

Ψ = np.zeros((S["store steps"] + 1, *([S["N"]] * 2)), dtype = np.cdouble)#csingle
            
m = 1 
    
if (S["imaginary time evolution"]):
    Ur = np.exp(-0.5*(dt/hbar)*V)
    Uk = np.exp(-0.5*(dt/(m*hbar))*p2)

else:
    Ur = np.exp(-0.5j*(dt/hbar)*V())
    Uk = np.exp(-0.5j*(dt/(m*hbar))*p2)
        
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()
    

tmp = pyfftw.zeros_aligned((S["N"], S["N"]), dtype='complex64',n = 16)
c = pyfftw.zeros_aligned((S["N"], S["N"]), dtype='complex64',n = 16)
          
print("store steps", S["store steps"])
print("Nt_per_store_step",Nt_per_store_step)

        
Ψ[0] = norm(𝜓0())       
phi = np.array([Ψ[0]])

# Define the ground state wave function
t0 = time.time()
bar = progressbar.ProgressBar(maxval=1)
for _ in bar(range(1)):
    ITEnp(phi, S["store steps"], Nt_per_store_step, Ur, Uk, True, S["imaginary time evolution"])
print("Took", time.time() - t0)


Ψ[0] = Ψ[-1]
phi = np.array([Ψ[0]])

nos = S["Number of States"]-1
if (nos):
    t0 = time.time()
    bar = progressbar.ProgressBar(maxval=nos)
    # raising operators
    for i in bar(range(nos)):
        ITEnp(phi, S["store steps"], Nt_per_store_step, Ur, Uk, True, S["imaginary time evolution"])
        phi = np.concatenate([phi, Ψ[-1][np.newaxis, :, :]], axis=0)
    print("Took", time.time() - t0)
        

def differentiate_twice(f):
    f = np.fft.ifft2(-p2*np.fft.fft2(f))
    return f

hbar = 1.054571817e-34    # Reduced Planck constant in J*s
m = 9.10938356e-31        # Mass of electron in kg
m_e = m

# Define the Hamiltonian operator
def hamiltonian_operator(psi):
    # Calculate the kinetic energy part of the Hamiltonian
    KE = -(hbar**2 / 2*m) * differentiate_twice(psi)
    # K = -(hbar^2 / 2m) * d^2/dx^2
    # KE = (hbar^2 / 2m) * |dpsi/dx|^2
    # Calculate the potential energy part of the Hamiltonian
    PE = V * psi
    # Combine the kinetic and potential energy parts to obtain the full Hamiltonian
    H = KE + PE
    return H


def expectation_value(psi, operator):
    operator_values = operator(psi)
    expectation = np.vdot(psi, operator_values)#E = <Ψ|H|Ψ> 
    return expectation

energies = np.array([expectation_value(i, hamiltonian_operator) for i in Ψ])

print("\nenergy =\n", energies.reshape(-1, 1))


Ψmax = np.amax(np.abs(Ψ))

Ψ_plot = Ψ/Ψmax

def complex_to_rgb(Z):
    """Convert complex values to their rgb equivalent.
    Parameters
    ----------
    Z : array_like
        The complex values.
    Returns
    -------
    array_like
        The rgb values.
    """
    #using HSV space
    r = np.abs(Z)
    arg = np.angle(Z)
    
    h = (arg + np.pi)  / (2 * np.pi)
    s = np.ones(h.shape)
    v = r  / np.amax(r)  #alpha
    c = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1)  ) # --> tuple
    return c


def complex_to_rgba(Z: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    r = np.abs(Z)
    arg = np.angle(Z)
    
    h = (arg + np.pi)  / (2 * np.pi)
    s = np.ones(h.shape)
    v = np.ones(h.shape)  #alpha
    rgb = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1)  ) # --> tuple

    abs_z = np.abs(Z)/ max_val
    abs_z = np.where(abs_z> 1., 1. ,abs_z)
    return np.concatenate((rgb, abs_z.reshape((*abs_z.shape,1))), axis= (abs_z.ndim))


def animate(xlim=None, ylim=None, figsize=(7, 7), animation_duration = 5, fps = 20, save_animation = False, 
    potential_saturation=0.8, title = "double slit experiment", wavefunction_saturation=0.8):
        
        total_frames = int(fps * animation_duration)
        
        px = 1 / plt.rcParams['figure.dpi']
        figsize = (640*px, 640*px)
        
        
        viridis = cm.get_cmap('gray', 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        mc = np.array([0, 43/256, 54/256, 1])

        newcolors[:150, :] = mc
        newcmp = ListedColormap(newcolors)
           
       
        fig = plt.figure(figsize=figsize, facecolor='#002b36')
      
        
        ax = fig.add_subplot(1, 1, 1)
        
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white') 
        
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)              
                

        L = S["extent"] / Å / 4
        potential_plot = ax.imshow((V + Vmin)/(Vmax-Vmin), 
        vmax = 1.0/potential_saturation, vmin = 0, cmap = newcmp, origin = "lower", 
        interpolation = "gaussian", extent = [-L/2, L/2, -L/2, L/2])
        

        wavefunction_plot = ax.imshow(complex_to_rgba(Ψ_plot[0], max_val= wavefunction_saturation),
        origin = "lower", interpolation = "gaussian", extent=[-L/2, L/2, -L/2, L/2])


        if xlim != None:
            ax.set_xlim(np.array(xlim)/Å)
        if ylim != None:
            ax.set_ylim(np.array(ylim)/Å)

        

        ax.set_title("$\psi(x,y,t)$"+" "+title, color = "white")
        ax.set_xlabel('[Å]')
        ax.set_ylabel('[Å]')

        time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top", alpha=0.9)
 

        xdt = np.linspace(0, S["total time"]/femtoseconds, total_frames)
        psi_index = np.linspace(0, S["store steps"], total_frames)
        
        def func_animation(frame):
            
            time_ax.set_text(u"t = {} femtoseconds".format("%.3f" % (xdt[frame])))
            index = int(psi_index[frame])
            wavefunction_plot.set_data(complex_to_rgba(Ψ_plot[index], max_val= wavefunction_saturation))
            
            return wavefunction_plot, time_ax


        ani = animation.FuncAnimation(fig, func_animation,
                                    blit=True, frames=total_frames, interval= 1/fps * 1000)
        if save_animation == True:
            if(title == ''):
                title = "animation"
            ani.save(S["path save"] + title +'.gif', fps = fps, metadata = dict(artist = 'Me'))
        else:
            plt.show()
            
            

animate(xlim=[-S["extent"]/8,S["extent"]/8], ylim=[-S["extent"]/8,S["extent"]/8], potential_saturation = 0.5, 
wavefunction_saturation = 0.2, animation_duration = S["animation duration"], 
fps = S["fps"], save_animation = S["save animation"], title=S["title"]+" "+str(S["Number of States"])+" eigenstates")

