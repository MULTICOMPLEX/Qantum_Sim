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


Å = 1.8897261246257702
femtoseconds = 4.134137333518212 * 10.
m_e = 1.0
hbar = 1.0

#extent = 25 * Å

n = 2048

js = {
 "name": "Q0",
 "mode": "two tunnel+-",
 "total time": 25 * femtoseconds,
 "store steps": 1000,
 "σ": 0.7 * Å, #0.2
 "beta2": 40, #initial_wavefunction momentum 
               #The constant is commonly represented as the second-order dispersion coefficient
 "beta2n": 10, #wavefunction momentum 
              
 "V0": 2, #barrier voltage 
 "initial offset": -70,
 "N": n,
 "dt": 2 / (np.log2(n) * np.sqrt(n)),
 "x0": 0, #barrier x
 "x1": 3,
 "x2": 12,
 "extent": 25 * Å,
 "extentN": -270 * Å,
 "extentP": +430 * Å,
 "imaginary time evolution": False,
 "animation duration": 8, #seconds
 "save animation": True,
 "fps": 30,
 "path save": "./gifs/",
 "title": "1D two potential barrier beta2 V2"
}

x = np.linspace(js["extentN"], js["extentP"], js["N"])


#interaction potential
def potential_barrier():
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0    
    a = 1 * Å 
    barrier = np.where(((x > js["x0"]* Å  - a/2) & (x < js["x0"]* Å  + a/2)), js["V0"], 0)
    barrier = np.where(((x > js["x1"]* Å  - a/2) & (x < js["x1"]* Å  + a/2)), -js["V0"], barrier)
    return barrier


#wavefunction at t = 0. 
def initial_wavefunction(offset = -15, v0 = 40):
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    offset = -offset
    v0 *= Å / femtoseconds
    p_x0 = m_e * v0 
    return np.exp( -1/(4* js["σ"]**2) * ((x + offset)**2) / np.sqrt(2*np.pi* js["σ"]**2))  *np.exp(p_x0*x*1j)

 
Vgrid = potential_barrier()
 
Vmin = np.amin(Vgrid)
Vmax = np.amax(Vgrid)

dx = x[1] - x[0]
px = np.fft.fftfreq(js["N"], d = dx) * hbar  * 2*np.pi
p2 = px**2 


Ψ = np.zeros((js["store steps"] + 1, * [js["N"]]), dtype = np.complex128)
dt_store = js["total time"]/js["store steps"]

Nt_per_store_step = int(np.round(dt_store / js["dt"]))
Nt_per_store_step = Nt_per_store_step

#time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
dt = dt_store/Nt_per_store_step

Ψ = np.zeros((js["store steps"] + 1, *([js["N"]])), dtype = np.complex128)
            
Ψ[0] = np.array(initial_wavefunction(js["initial offset"], js["beta2"]))
 
m = 1     
Ur = np.exp(-0.5j*(dt/hbar)*np.array(Vgrid)) 
Uk = np.exp(-0.5j*(dt/(m*hbar))*p2) 
        
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()
     
tmp = pyfftw.empty_aligned(js["N"],  dtype='complex128')
c = pyfftw.empty_aligned(js["N"], dtype='complex128')
fft_object = pyfftw.FFTW(tmp, c, direction='FFTW_FORWARD')
ifft_object = pyfftw.FFTW(c, tmp, direction='FFTW_BACKWARD')
         
print("store_steps",js["store steps"])
print("Nt_per_store_step",Nt_per_store_step)
        
#c = np.fft.fftn(Ur*tmp)
#tmp = Ur*np.fft.ifftn(Uk*c)

v0 = js["beta2n"]
v0 *= Å / femtoseconds
p_x0 = m_e * v0 

k = np.exp(p_x0*x*1j)
k /= np.amax(k)
        
t0 = time.time()
bar = progressbar.ProgressBar()
for i in bar(range(js["store steps"])):
    tmp = np.copy(Ψ[i])
    if(i==100):
        tmp = np.abs(tmp) * k
    for j in range(Nt_per_store_step):
           fft_object(Ur*tmp, c)
           tmp = Ur * ifft_object(Uk*c, tmp)
    Ψ[i+1] = tmp

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
               

        plt.gcf().axes[0].axvspan(js["x0"] - 0.5, js["x0"] + 0.5, alpha=0.5, color='red')
        plt.text(js["x0"] - 7., 0.95, '+', horizontalalignment='center', verticalalignment = 'center', color = 'red')
        plt.gcf().axes[0].axvspan(js["x1"] - 0.5, js["x1"] + 0.5, alpha=0.5, color='gray')
        plt.text(js["x1"] + 7., 0.95, '-', horizontalalignment='center', verticalalignment = 'center', color = 'gray')
        
        time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (0./femtoseconds)))

        plt.ylim(-1, 1)

        index = 0
        

        real_plot, = ax.plot(x/Å, np.real(Ψ[index]), label='$Re|\psi(x)|$')
        imag_plot, = ax.plot(x/Å, np.imag(Ψ[index]), label='$Im|\psi(x)|$')
        abs_plot, = ax.plot(x/Å, np.abs(Ψ[index]), label='$|\psi(x)|$')
        
        
        ax.set_facecolor('#002b36')
        
        leg = ax.legend(facecolor='#002b36',loc = 'lower left')
        for line, text in zip(leg.get_lines(), leg.get_texts()):
               text.set_color(line.get_color())


        #print(total_frames)
        animation_data = {'t': 0.0, 'x' :x, 'ax':ax ,'frame' : 0}
        def func_animation(*arg):
            
            time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (animation_data['t']/femtoseconds)))

            animation_data['t'] = animation_data['t'] + dt
            if animation_data['t'] > js["total time"]:
                animation_data['t'] = 0.0

            #print(animation_data['frame'])
            animation_data['frame'] +=1
            index = int((js["store steps"])/js["total time"] * animation_data['t'])

            real_plot.set_ydata(np.real(Ψ[index]))
            imag_plot.set_ydata(np.imag(Ψ[index]))
            abs_plot.set_ydata(np.abs(Ψ[index]))


            return real_plot,imag_plot,abs_plot, time_ax

        frame = 0
        ani = animation.FuncAnimation(fig, func_animation,
                                    blit=False, frames=total_frames, interval= 1/fps * 1000)
        if save_animation == True:
            if(title == ''):
                title = "animation"
            ani.save(js["path save"] + title +'.gif', fps = fps, metadata = dict(artist = 'Me'))
        else:
            plt.show()


#visualize the time dependent simulation
animate(animation_duration = js["animation duration"], fps = js["fps"], save_animation = js["save animation"], title = js["title"])

