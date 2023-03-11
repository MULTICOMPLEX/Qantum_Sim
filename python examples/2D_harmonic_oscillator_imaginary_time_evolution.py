import numpy as np
import time
import progressbar
import multiprocessing
from visuals import *
from constants import *
from functions import *
from scipy.stats import multivariate_normal

S = {
 "total time": 10 * const["femtoseconds"], #for V_Coulomb: 15, NStates >= 5  = 20, >= 8 =35
 "extent": 30 * const["Å"], #30
 "N": 350,
 "store steps": 20,
 "dt":  1,
 "Number of States": 5,
 "imaginary time evolution": True,
 "animation duration": 4, #seconds
 "save animation": True,
 "fps": 30,
 "path save": "./gifs/",
 "σ_x": 1. * const["Å"], 
 "σ_y": 1. * const["Å"], 
 "v0_x": 64. * const["Å"] / const["femtoseconds"], #initial_wavefunction momentum x
 "v0_y": 64. * const["Å"] / const["femtoseconds"], #initial_wavefunction momentum y
 "initial wavefunction offset x": 0 * const["Å"],
 "initial wavefunction offset y": 0 * const["Å"], 
 "title": "2D harmonic oscillator Coulomb potential" #rotationally symmetric
 #"title": "2D harmonic oscillator rotationally symmetric potential"
}
   
X = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])
Y = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])
dx = X[1] - X[0]
dy = Y[1] - Y[0]
X, Y = np.meshgrid(X,Y)


#potential energy operator
# rotationally symmetric potential
#V(x,y) = (1/2) k (x^2 + y^2)
def V_Rotational(X, Y):
    k = 0.5
    return k * (X**2 + Y**2)

#potential energy operator
# Coulomb potential for a point charge
#V(x,y) = (q/4πε) log[(x^2 + y^2)^(1/2)]
def V_Coulomb(X, Y):
    q = 0.5
    ε = 1
    π = np.pi
    return (q/4*π*ε) * np.log(np.power((X**2 + Y**2),0.5))


V = V_Coulomb(X, Y) 
Vmin = np.amin(V)
Vmax = np.amax(V)
    

#initial waveform
def 𝜓0_gaussian_wavepacket_2D1(X, Y, v0_x, v0_y, sigma_x, sigma_y, x0, y0):
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    p_x0 = const["m_e"] * v0_x
    p_y0 = const["m_e"] * v0_y
    σ = sigma_x
    
    norm = 1 / (2 * np.pi * sigma_x * sigma_y)
    Z = np.exp(-(X-x0)**2/(2*sigma_x**2) - (Y-y0)**2/(2*sigma_y**2)) * norm * np.exp(1j*(p_x0*X+p_y0*Y))
    #Zmax = np.amax(np.abs(Z))
    #Z = Z/Zmax

    #return Z 
    
    return np.exp( -1/(4* σ**2) * ((X-x0)**2 + (Y-y0)**2)) / np.sqrt(2*np.pi* σ**2) * np.exp(1j*(p_x0*X+p_y0*Y))     
    #return np.exp(-(X-x0)**2/(4*σ_x**2) - (Y-y0)**2/(4*σ_y**2)) * np.exp(1j*(p_x0*X + p_y0*Y))
    #np.exp(1j*(p_x0*X*Y+p_y0*Y*X)) 
    #np.exp(1j*(p_x0*X**2+p_y0*Y**2)) 


#initial waveform
def 𝜓0_gaussian_wavepacket_2D(X, Y, v0_x, v0_y, sigma_x, sigma_y, x0, y0):
    p_x0 = const["m_e"] * v0_x
    p_y0 = const["m_e"] * v0_y
    mean = [x0, y0]
    cov = [[4, 0], [0, 4]]
    pos = np.dstack((X, Y))
    # Create the multivariate normal distribution object
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos) 
    Zmax = np.amax(np.abs(Z))
    Z /= Zmax 
    Z = Z * np.exp(1j*(p_x0*X + p_y0*Y))
    return Z 


psi_0 = 𝜓0_gaussian_wavepacket_2D(X, Y, S["v0_x"], 0, S["σ_x"], S["σ_y"], S["initial wavefunction offset x"], 
S["initial wavefunction offset y"])

#complex_plot_2D(psi_0, S["extent"], V, Vmin, Vmax) ;exit()


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
        

tmp = pyfftw.zeros_aligned((S["N"], S["N"]), dtype='complex64',n = 16)
c = pyfftw.zeros_aligned((S["N"], S["N"]), dtype='complex64',n = 16)
          
print("Number of States", S["Number of States"])
print("store steps", S["store steps"])
print("Nt_per_store_step",Nt_per_store_step)
        
Ψ[0] = norm(psi_0, dx)       
phi = np.array([Ψ[0]])

# Define the ground state wave function
t0 = time.time()
bar = progressbar.ProgressBar(maxval=1)
for _ in bar(range(1)):
    Split_Step_NP(Ψ, phi, dx, S["store steps"], Nt_per_store_step, Ur, Uk, S["imaginary time evolution"])
print("Took", time.time() - t0)


Ψ[0] = Ψ[-1]
phi = np.array([Ψ[0]])

nos = S["Number of States"]-1
if (nos):
    t0 = time.time()
    bar = progressbar.ProgressBar(maxval=nos)
    # raising operators
    for i in bar(range(nos)):
        Split_Step_NP(Ψ, phi, dx, S["store steps"], Nt_per_store_step, Ur, Uk, S["imaginary time evolution"])
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