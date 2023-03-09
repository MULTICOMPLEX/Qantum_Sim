import numpy as np
import time
import progressbar
import multiprocessing
from visuals import *
from constants import *
from functions import *

S = {
 "total time": 10 * const["femtoseconds"], #for V_Coulomb: 15, NStates >= 5  = 20, >= 8 =35
 "extent": 30 * const["Å"], #30
 "N": 350,
 "store steps": 20,
 "dt":  0.5,
 "Number of States": 4,
 "imaginary time evolution": True,
 "animation duration": 4, #seconds
 "save animation": True,
 "fps": 30,
 "path save": "./gifs/",
 "σ_x": 1. * const["Å"], 
 "σ_y": 1. * const["Å"], 
 "v0": 64. * const["Å"] / const["femtoseconds"], #initial_wavefunction momentum #64
 "initial wavefunction offset x": 0 * const["Å"],
 "initial wavefunction offset y": 0 * const["Å"], 
 "title": "2D harmonic oscillator Coulomb potential" #rotationally symmetric
 #"title": "2D harmonic oscillator rotationally symmetric potential"
}
   
x = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])
y = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])
dx = x[1] - x[0]
dy = y[1] - y[0]
x, y = np.meshgrid(x,y)


#potential energy operator
# rotationally symmetric potential
#V(x,y) = (1/2) k (x^2 + y^2)
def V_Rotational():
    k = 0.5
    return k * (x**2 + y**2)

#potential energy operator
# Coulomb potential for a point charge
#V(x,y) = (q/4πε) log[(x^2 + y^2)^(1/2)]
def V_Coulomb():
    q = 0.5
    ε = 1
    π = np.pi
    return (q/4*π*ε) * np.log(np.power((x**2 + y**2),0.5))


V = V_Coulomb() 
Vmin = np.amin(V)
Vmax = np.amax(V)
    
#initial waveform
def 𝜓0_x():
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    p_x0 = const["m_e"] * S["v0"]
    σ = S["σ_x"]
    return np.exp( -1/(4* σ**2) * ((x-S["initial wavefunction offset x"])**2+
    (y-S["initial wavefunction offset y"])**2)) / np.sqrt(2*np.pi* σ**2)  *np.exp(p_x0*x*1j)
 
#initial waveform
def 𝜓0_y():
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_y0
    p_y0 = m_e * S["v0"]
    σ = S["σ_y"]
    return np.exp( -1/(4* σ**2) * ((x-S["initial wavefunction offset x"])**2+
    (y-S["initial wavefunction offset y"])**2)) / np.sqrt(2*np.pi* σ**2)  *np.exp(p_y0*y*1j) 


#plot(𝜓0_x(), S["extent"], V, Vmin, Vmax)


dt_store = S["total time"] / S["store steps"]
Nt_per_store_step = int(np.round(dt_store / S["dt"]))
#time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
dt = dt_store/Nt_per_store_step

Ψ = np.zeros((S["store steps"] + 1, *([S["N"]] * 2)), dtype = np.cdouble)#csingle
            
            
p2 = fft_frequencies(S["N"], dx, const["hbar"])
   
if (S["imaginary time evolution"]):
    Ur = np.exp(-0.5*(dt/const["hbar"])*V)
    Uk = np.exp(-0.5*(dt/(const["m"]*const["hbar"]))*p2)

else:
    Ur = np.exp(-0.5j*(dt/const["hbar"])*V())
    Uk = np.exp(-0.5j*(dt/(const["m"]*const["hbar"]))*p2)
        
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()
    

tmp = pyfftw.zeros_aligned((S["N"], S["N"]), dtype='complex64',n = 16)
c = pyfftw.zeros_aligned((S["N"], S["N"]), dtype='complex64',n = 16)
          
print("store steps", S["store steps"])
print("Nt_per_store_step",Nt_per_store_step)

        
Ψ[0] = norm(𝜓0_x(), dx)       
phi = np.array([Ψ[0]])

# Define the ground state wave function
t0 = time.time()
bar = progressbar.ProgressBar(maxval=1)
for _ in bar(range(1)):
    Split_Step_FFTW(Ψ, phi, dx, S["store steps"], Nt_per_store_step, Ur, Uk, S["imaginary time evolution"])
print("Took", time.time() - t0)


Ψ[0] = Ψ[-1]
phi = np.array([Ψ[0]])

nos = S["Number of States"]
if (nos):
    t0 = time.time()
    bar = progressbar.ProgressBar(maxval=nos)
    # raising operators
    for i in bar(range(nos)):
        Split_Step_FFTW(Ψ, phi, dx, S["store steps"], Nt_per_store_step, Ur, Uk, S["imaginary time evolution"])
        phi = np.concatenate([phi, [Ψ[-1]]])
    print("Took", time.time() - t0)
        

hbar = 1.054571817e-34    # Reduced Planck constant in J*s
m = 9.10938356e-31        # Mass of electron in kg
m_e = m

# Define the Hamiltonian operator
def hamiltonian_operator(psi):
    # Calculate the kinetic energy part of the Hamiltonian
    KE = -(hbar**2 / 2*m) * differentiate_twice(psi, p2)
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


title = ""

if(S["Number of States"]==0):
  title = title=S["title"]+ " Ground state"

if(S["Number of States"]==1):
  title = title=S["title"]+" "+str(S["Number of States"])+"st eigenstate"

if(S["Number of States"]==2):
  title = title=S["title"]+" "+str(S["Number of States"])+"nd eigenstate"

if(S["Number of States"]==3):
  title = title=S["title"]+" "+str(S["Number of States"])+"rd eigenstate" 

if(S["Number of States"]>=4):
  title = title=S["title"]+" "+str(S["Number of States"])+"th eigenstate"

ani = animate(
Ψ_plot, 
energies, 
S["extent"], 
V, 
Vmin, 
Vmax, 
xlim=[-S["extent"]/8, S["extent"]/8], 
ylim=[-S["extent"]/8, S["extent"]/8], 
potential_saturation = 0.5, 
wavefunction_saturation = 0.2, 
animation_duration = S["animation duration"], 
fps = S["fps"], 
save_animation = S["save animation"], 
title=title, 
path_save = S["path save"], 
total_time = S["total time"], 
store_steps = S["store steps"])