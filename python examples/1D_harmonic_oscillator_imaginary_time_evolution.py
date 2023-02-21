import numpy as np
import time
import progressbar
import pyfftw
import scipy
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb
import copy


Å = 1.8897261246257702
femtoseconds = 4.134137333518212 * 10.
m_e = 1.0
hbar = 1.0

#extent = 25 * Å

n = 2048

js = {
 "name": "Q0",
 "mode": "two tunnel+-",
 "total time": 1 * femtoseconds, 
 "store steps": 20,
 "σ": 0.7 * Å, 
 "v0": 60, #initial_wavefunction momentum
 "V0": 2, #barrier voltage 
 "initial offset": 0,
 "N": n,
 "dt": 0.5, 
 "x0": 0, #barrier x
 "x1": 3,
 "x2": 12,
 "extent": 30 * Å,#150, 30
 "extentN": -75 * Å,
 "extentP": +85 * Å,
 "NW": 150, #+1 ground state
 "imaginary time evolution": True,
 "animation duration": 10, #seconds
 "save animation": False,
 "fps": 30,
 "path save": "./gifs/",
 "title": "1D harmonic oscillator imaginary time evolution"
}

x = np.linspace(-js["extent"]/2, js["extent"]/2, js["N"])

#interaction potential
def harmonic_oscillator():
    m = m_e
    T = 0.6*femtoseconds
    w = 2*np.pi/T
    k = m* w**2
    return 1.6 * k * x**2 #150, 1

#=========================================================================================================#
# Define the wavefunction at t = 0  (initial condition)
#=========================================================================================================#

def initial_wavefunction():
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = 0.7 * Å
    v0 = js["v0"] * Å / femtoseconds
    p_x0 = m_e * v0
    return np.exp( -1/(4* σ**2) * ((x-js["initial offset"])**2) / np.sqrt(2*np.pi* σ**2))  *np.exp(p_x0*x*1j)


def norm(phi):
    norm = np.sum(np.square(np.abs(phi)))*dt
    return phi/np.sqrt(norm)
    

def apply_projection(tmp, psi_list):
    for psi in psi_list:
        tmp -= np.sum(tmp*np.conj(psi)*dt)*psi
    return tmp

def ITE(phi, store_steps, Nt_per_store_step, Ur, Uk, tmp):
    for i in range(store_steps):
        tmp = np.copy(Ψ[i])
        for j in range(Nt_per_store_step):
            fft_object(Ur*tmp, c)
            ifft_object(Uk*c, tmp)
            tmp *= Ur
            tmp = norm(apply_projection(tmp, phi))
        Ψ[i+1] = tmp
    return 

def ITEnp(phi, store_steps, Nt_per_store_step, Ur, Uk, tmpp):
    for i in range(store_steps):
        tmp = Ψ[i]
        for j in range(Nt_per_store_step):
            c = np.fft.fftn(Ur*tmp)
            tmp = Ur * np.fft.ifftn(Uk*c)
            tmp = norm(apply_projection(tmp, phi))
        Ψ[i+1] = tmp
    return 

def complex_plot(phi):
    plt.plot(x, np.abs(phi))
    plt.plot(x, np.real(phi))
    plt.plot(x, np.imag(phi))
    plt.show()
    return 

Vgrid = harmonic_oscillator()
 
Vmin = np.amin(Vgrid)
Vmax = np.amax(Vgrid)

dx = x[1] - x[0]
px = np.fft.fftfreq(js["N"], d = dx) * hbar  * 2*np.pi
p2 = px**2


dt_store = js["total time"]/js["store steps"]
Nt_per_store_step = int(np.round(dt_store / js["dt"]))

dt = dt_store/Nt_per_store_step

Ψ = np.zeros((js["store steps"] + 1, *([js["N"]])), dtype = np.complex128)
            
m = 1     
if(js["imaginary time evolution"]):     
    Ur = np.exp(-0.5*(dt/hbar)*Vgrid)
    Uk = np.exp(-0.5*(dt/(m*hbar))*p2)

else:
  Ur = np.exp(-0.5j*(dt/hbar)*Vgrid)
  Uk = np.exp(-0.5j*(dt/(m*hbar))*p2)
        
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()


tmp = pyfftw.empty_aligned(js["N"],  dtype='complex128')
c = pyfftw.empty_aligned(js["N"], dtype='complex128')
fft_object = pyfftw.FFTW(Ur * tmp, c, direction='FFTW_FORWARD')
ifft_object = pyfftw.FFTW(c, tmp, direction='FFTW_BACKWARD')

         
print("store_steps",js["store steps"])
print("Nt_per_store_step",Nt_per_store_step)

Ψ[0] = norm(initial_wavefunction())
phi = [Ψ[0]]

# Define the ground state wave function
t0 = time.time()
bar = progressbar.ProgressBar(maxval=1)
for i in bar(range(js["NW"])):
    ITE(phi, js["store steps"], Nt_per_store_step, Ur, Uk, tmp)
print("Took", time.time() - t0)

Ψ[0] = norm(Ψ[-1])
phi = [Ψ[0]]

t0 = time.time()
bar = progressbar.ProgressBar(maxval=js["NW"])
#raising operators
for i in bar(range(js["NW"])):
    ITE(phi, js["store steps"], Nt_per_store_step, Ur, Uk, tmp)
    phi.append(norm(Ψ[-1])) 
print("Took", time.time() - t0)


Ψ /= np.amax(np.abs(Ψ))

def animate(xlim=None, figsize=(16/9 *5.804 * 0.9, 5.804), animation_duration = 5, fps = 20, save_animation = False, 
    title = "1D potential barrier"):
    
        total_frames = int(fps * animation_duration)
        
       
        dt = js["total time"]/total_frames

        fig = plt.figure(figsize=figsize, facecolor='#002b36')
        ax = fig.add_subplot(1, 1, 1)
        index = 0
        
        
        ax.set_xlabel("[Å]")
        ax.set_title("$\psi(x,t)$"+" "+title, color = 'white')
        
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white') 
               
        
        time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (0./femtoseconds)))

        plt.xlim(xlim)
        plt.ylim(-1, 1.1)
        

        index = 0
        
        potential_plot = ax.plot(x/Å, (Vgrid + Vmin)/(Vmax-Vmin), label='$V(x)$')  
        real_plot, = ax.plot(x/Å, np.real(Ψ[index]), label='$Re|\psi(x)|$')
        imag_plot, = ax.plot(x/Å, np.imag(Ψ[index]), label='$Im|\psi(x)|$')
        abs_plot, = ax.plot(x/Å, np.abs(Ψ[index]), label='$|\psi(x)|$')
        
        
        ax.set_facecolor('#002b36')
        
        leg = ax.legend(facecolor='#002b36',loc = 'lower left')
        for line, text in zip(leg.get_lines(), leg.get_texts()):
               text.set_color(line.get_color())
        

        animation_data = {'t': 0.0, 'x' :x, 'ax':ax ,'frame' : 0, 'index' : 0}
        def func_animation(*arg):
            
            time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (animation_data['t']/femtoseconds)))

            if animation_data['t'] > js["total time"]:
                animation_data['t'] = 0.0

            index = animation_data['index']
         
            #print(index)
            
            real_plot.set_ydata(np.real(Ψ[index]))
            imag_plot.set_ydata(np.imag(Ψ[index]))
            abs_plot.set_ydata(np.abs(Ψ[index]))
            
            animation_data['index'] = int((js["store steps"])/js["total time"] * animation_data['t'])
            animation_data['frame'] +=1
            animation_data['t'] += dt
            
            return 

      
        ani = animation.FuncAnimation(fig, func_animation,
                                    blit=False, frames=total_frames, interval= 1/fps * 1000, cache_frame_data = False)
        if save_animation == True:
            if(title == ''):
                title = "animation"
            ani.save(js["path save"] + title +'.gif', fps = fps, metadata = dict(artist = 'Me'))
        else:
            plt.show()


#visualize the time dependent simulation
animate(xlim=[-js["extent"]/2/Å, js["extent"]/2/Å], animation_duration = js["animation duration"], fps = js["fps"], 
save_animation = js["save animation"], title = js["title"]+" "+str(js["NW"]+1)+" cycles")

