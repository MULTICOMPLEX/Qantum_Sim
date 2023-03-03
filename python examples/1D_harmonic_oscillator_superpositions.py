import numpy as np
import time
import progressbar
import pyfftw
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib import animation
from visuals import *
from constants import *

n = 2048*2

S = {
    "name": "Q0",
    "mode": "two tunnel+-",
    "total time": 0.5 * femtoseconds,
    "store steps": 20,
    "σ": 0.5 * Å,
    "v0": 60,  # T momentum
    "V0": 2,  # barrier voltage
    "initial offset": 0,
    "N": n,
    "dt": 0.25,
    "x0": 0,  # barrier x
    "x1": 3,
    "x2": 12,
    "extent": 20 * Å,  # 150, 30
    "extentN": -75 * Å,
    "extentP": +85 * Å,
    "NW": 30,
    "imaginary time evolution": True,
    "animation duration": 10,  # seconds
    "save animation": False,
    "fps": 30,
    "path save": "./gifs/",
    "title": "1D harmonic oscillator superpositions"
}

x = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])

#potential energy operator
def V():
    return 2 * m_e * (2 * np.pi / (0.6 * femtoseconds))**2 * x**2

#initial waveform
def 𝜓0(σ, v0, offset):
    # This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = σ
    v0 = v0 * Å / femtoseconds
    p_x0 = m_e * v0
    return np.exp(-1/(4 * σ**2) * ((x-offset)**2) / np.sqrt(2*np.pi * σ**2)) * np.exp(p_x0*x*1j)

V = V()
Vmin = np.amin(V)
Vmax = np.amax(V)

dx = x[1] - x[0]
px = np.fft.fftfreq(S["N"], d=dx) * hbar * 2*np.pi
p2 = px**2


dt_store = S["total time"]/S["store steps"]
Nt_per_store_step = int(np.round(dt_store / S["dt"]))

dt = dt_store/Nt_per_store_step

Ψ = np.zeros((S["store steps"] + 1, *([S["N"]])), dtype=np.complex128)

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


tmp = pyfftw.empty_aligned(S["N"],  dtype='complex128')
c = pyfftw.empty_aligned(S["N"], dtype='complex128')
fft_object = pyfftw.FFTW(Ur * tmp, c, direction='FFTW_FORWARD')
ifft_object = pyfftw.FFTW(c, tmp, direction='FFTW_BACKWARD')

def ITE(Ψ, phi, dt, store_steps, Nt_per_store_step, Ur, Uk, tmp):
    for i in range(store_steps):
        tmp = np.copy(Ψ[i])
        for _ in range(Nt_per_store_step):
            fft_object(Ur*tmp, c)
            ifft_object(Uk*c, tmp)
            tmp *= Ur
            tmp = apply_projection(tmp, phi, dt)
            # tmp = norm(apply_projection(tmp, phi))
        Ψ[i+1] = norm(tmp, dt)
    return

print("store_steps", S["store steps"])
print("Nt_per_store_step", Nt_per_store_step)

Ψ[0] = norm(𝜓0(S["σ"], S["v0"], S["initial offset"]), dt)
phi = np.array([Ψ[0]])

# Define the ground state wave function
t0 = time.time()
bar = progressbar.ProgressBar(maxval=1)
for _ in bar(range(1)):
    ITE(Ψ, phi, dt, S["store steps"], Nt_per_store_step, Ur, Uk, tmp)
print("Took", time.time() - t0)

Ψ[0] = Ψ[-1]
phi = np.array([Ψ[0]])

t0 = time.time()
if (S["NW"]-1):
    bar = progressbar.ProgressBar(maxval=S["NW"])
# raising operators
for _ in bar(range(S["NW"]-1)):
    ITE(Ψ, phi, dt, S["store steps"], Nt_per_store_step, Ur, Uk, tmp)
    phi = np.vstack([phi, Ψ[-1]])
if (S["NW"]-1):
    print("Took", time.time() - t0)


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

energies = np.array([expectation_value(i, hamiltonian_operator) for i in phi])
#print("\energies =\n", energies.reshape(-1, 1))

eigenstates = np.array(phi) 

𝜓0 = norm(𝜓0(S["σ"], S["v0"], S["initial offset"]), dt)

coeffs = np.dot(eigenstates.conj(), 𝜓0)

initial_waveform = np.dot(eigenstates.T, coeffs)
complex_plot(x, initial_waveform)

superpositions(eigenstates, coeffs, energies, extent=10*Å, save_animation = S["save animation"])

Ψ /= np.max(np.abs(Ψ))


def animate(xlim=None, figsize=(16/9 * 5.804 * 0.9, 5.804), animation_duration=5, fps=20, save_animation=False,
            title="1D potential barrier"):

    total_frames = int(fps * animation_duration)


    fig = plt.figure(figsize=figsize, facecolor='#002b36')
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("[Å]")
    ax.set_title("$\psi(x,t)$"+" "+title, color='white')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    time_ax = ax.text(0.97, 0.97, "",  color="white",
                      transform=ax.transAxes, ha="right", va="top")
                      
    energy_ax = ax.text(0.97, 0.07, "", color='white',
                        transform=ax.transAxes, ha="right", va="top")

    plt.xlim(xlim)
    plt.ylim(-1, 1.1)

    index = 0
    
    potential_plot = ax.plot(x/Å, (V + Vmin)/(Vmax-Vmin), label='$V(x)$')
    real_plot, = ax.plot(x/Å, np.real(Ψ[index]), label='$Re|\psi(x)|$')
    imag_plot, = ax.plot(x/Å, np.imag(Ψ[index]), label='$Im|\psi(x)|$')
    abs_plot, = ax.plot(x/Å, np.abs(Ψ[index]), label='$|\psi(x)|$')

    ax.set_facecolor('#002b36')

    leg = ax.legend(facecolor='#002b36', loc='lower left')
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    

    xdt = np.linspace(0, S["total time"]/femtoseconds, total_frames)
    psi_index = np.linspace(0, S["store steps"], total_frames)
    
    def func_animation(frame):

        index = int(psi_index[frame])
        
        time_ax.set_text(u"t = {} femtoseconds".format(
            "%.3f" % (xdt[frame])))
        
        real_plot.set_ydata(np.real(Ψ[index]))
        imag_plot.set_ydata(np.imag(Ψ[index]))
        abs_plot.set_ydata(np.abs(Ψ[index]))
        
        return

    ani = animation.FuncAnimation(fig, func_animation,
                                  blit=False, frames=total_frames, interval=1/fps * 1000, cache_frame_data=False)
    if save_animation == True:
        if (title == ''):
            title = "animation"
        ani.save(S["path save"] + title + '.gif',
                 fps=fps, metadata=dict(artist='Me'))
    else:
        plt.show()


# visualize the time dependent simulation
animate(xlim=[-S["extent"]/2/Å, S["extent"]/2/Å], animation_duration=S["animation duration"], fps=S["fps"],
        save_animation=S["save animation"], title=S["title"]+" "+str(S["NW"])+" states")
