import numpy as np
import time
import progressbar
import pyfftw
import scipy
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import animation
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb


Å = 1.8897261246257702
femtoseconds = 4.134137333518212 * 10.
hbar = 1.0

n = 256

js = {
 "total time": 0.5 * femtoseconds,
 "store steps": 200,
 "σ": 0.4 * Å, #initial_wavefunction
 
 "𝜇01": -5.0 * Å, #fermions
 "𝜇02": 2.0 * Å, #fermions
 
 #"𝜇01": -7.0 * Å, #bosons
 #"𝜇02": 0.0 * Å, #bosons
 
 "N": n,
 "dt": 4 / (np.log2(n) * np.sqrt(n)),
 "extent": 25 * Å,
 "animation duration": 10, #seconds
 "save animation": False,
 "path save": "./gifs/",
 "title": "1D interactive bosons"
}


x1 = np.linspace(-js["extent"]/2, js["extent"]/2, js["N"])
x2 = np.linspace(-js["extent"]/2, js["extent"]/2, js["N"])
x1, x2 = np.meshgrid(x1, x2)

def harmonic_oscillator_plus_coulomb_interaction():
    k = 0.5
    V_harmonic = 0.5*k*x1**2 + 0.5*k*x2**2
    k = 30.83
    r = np.abs(x1 - x2)
    r = np.where(r < 0.0001, 0.0001, r)
    V_coulomb_interaction = k / r

    return V_harmonic + V_coulomb_interaction
    
    
def initial_wavefunction():
    #This wavefunction correspond to two stationary gaussian wavepackets. The wavefunction must be symmetric: Ψ(x1,x2) = Ψ(x2,x1)
    σ = js["σ"]
    𝜇01 = js["𝜇01"]
    𝜇02 = js["𝜇02"]

    return (np.exp(-(x1 - 𝜇01)**2/(4*σ**2))*np.exp(-(x2 - 𝜇02)**2/(4*σ**2)) 
            + np.exp(-(x1 - 𝜇02)**2/(4*σ**2))*np.exp(-(x2 - 𝜇01)**2/(4*σ**2)))
    

Vgrid = harmonic_oscillator_plus_coulomb_interaction() 
Vmin = np.amin(Vgrid)
Vmax = np.amax(Vgrid)

dx = x1[0][1] - x1[0][0]
p1 = np.fft.fftfreq(js["N"], d = dx) * hbar  * 2*np.pi
p2 = np.fft.fftfreq(js["N"], d = dx) * hbar  * 2*np.pi
p1, p2 = np.meshgrid(p1, p2)
p2 = (p1**2 + p2**2)

        
store_steps = 200
dt_store = js["total time"]/store_steps

Nt_per_store_step = int(np.round(dt_store / js["dt"]))
Nt_per_store_step = Nt_per_store_step

#time/dt and dt_store/dt must be integers. Otherwise dt is rounded to match that the Nt_per_store_stepdivisions are integers
dt = dt_store/Nt_per_store_step

Ψ = np.zeros((store_steps + 1, *([js["N"]] * 2)), dtype = np.complex128)
            
Ψ[0] = np.array(initial_wavefunction())

m = 1   
Ur = np.exp(-0.5j*(dt/hbar)*Vgrid)
Uk = np.exp(-0.5j*(dt/(m*hbar))*p2)
        
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()
scipy.fft.set_backend(pyfftw.interfaces.scipy_fft)
    
           
tmp = pyfftw.empty_aligned((js["N"], js["N"]), dtype='complex128')
c = pyfftw.empty_aligned((js["N"], js["N"]), dtype='complex128')
fft_object = pyfftw.FFTW(tmp, c, direction='FFTW_FORWARD', axes=(0,1))
ifft_object = pyfftw.FFTW(c, tmp, direction='FFTW_BACKWARD', axes=(0,1))
           
        
print("store_steps",store_steps)
print("Nt_per_store_step",Nt_per_store_step)
        
        
t0 = time.time()
bar = progressbar.ProgressBar()
for i in bar(range(store_steps)):
    tmp = np.copy(Ψ[i])
    for j in range(Nt_per_store_step):
           fft_object(Ur*tmp, c)
           tmp = Ur * ifft_object(Uk*c, tmp)
    Ψ[i+1] = tmp

print("Took", time.time() - t0)
        

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


viridis = cm.get_cmap('gray', 256)
newcolors = viridis(np.linspace(0, 1, 256))
mc = np.array([0, 43/256, 54/256, 1])
newcolors[:255, :] = mc
newcmp = ListedColormap(newcolors)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[:50, :] = mc
newcmp2 = ListedColormap(newcolors)

def plot(t, xlim=None, figsize=(10, 5), potential_saturation=0.8, wavefunction_saturation=1.0, title = ""):


        fig = plt.figure(figsize=figsize, facecolor='#002b36') 
        
       
        grid = plt.GridSpec(10, 10, hspace=0.4, wspace=6.0)
        ax1 = fig.add_subplot(grid[0:10, 0:5])
        ax2 = fig.add_subplot(grid[3:7, 5:10], sharex=ax1) # probability density of finding any particle at x 

        ax1.set_xlabel("$x_1$ [Å]")
        ax1.set_ylabel("$x_2$ [Å]")
        ax1.set_title("$\psi(x_1,x_2)$"+" "+title, color = "white")

        ax2.set_xlabel("$x$ [Å]")
        ax2.set_ylabel("${\| \Psi(x)\|}^{2} $")
        ax2.set_title("Probability density", color = "white")


        time_ax = ax2.text(0.97,0.97, "",  color = "white",
                        transform=ax2.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format("%.2f"  % (t/femtoseconds)))



        if xlim != None:
            ax1.set_xlim(np.array(xlim)/Å)
            ax1.set_ylim(np.array(xlim)/Å)
            ax2.set_xlim(np.array(xlim)/Å)


        index = int((store_steps)/ js["total time"] *t)
        
        L = js["extent"] / Å
       
           
        
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.tick_params(colors='white')
        ax1.spines['left'].set_color('white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['right'].set_color('white') 
        
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['top'].set_linewidth(1)
        ax1.spines['right'].set_linewidth(1)

        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.tick_params(colors='white')
        ax2.spines['left'].set_color('white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white') 
        
        ax2.spines['left'].set_linewidth(1)
        ax2.spines['bottom'].set_linewidth(1)
        ax2.spines['top'].set_linewidth(1)
        ax2.spines['right'].set_linewidth(1)   
        
        
        ax1.imshow((Vgrid + Vmin)/(Vmax-Vmin), 
        vmax = 1.0/potential_saturation, vmin = 0, cmap = newcmp2, origin = "lower", interpolation = "bilinear", 
        extent = [-L/2, L/2, -L/2, L/2])  

        ax1.imshow(complex_to_rgba(Ψ_plot[index], max_val= wavefunction_saturation), origin = "lower", 
        interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  
        
        prob_density = np.abs(np.sum(  (Ψ_plot[index])*np.conjugate(Ψ_plot[index])  , axis = 1))
        x = np.linspace(-L/2, L/2, js["N"])
        prob_plot = ax2.plot(x,  prob_density, color= "cyan")
        prob_plot_fill = ax2.fill_between(x,prob_density, alpha=0.1, color= "cyan" )
        #ax1.set_aspect('equal', adjustable='box')
        #ax2.set_aspect('equal', adjustable='box')
        
        ax2.set_facecolor('#002b36')

        ax2.set_ylim([0,np.amax(prob_density)*1.3])


        plt.show()


def animate(xlim=None, ylim=None, figsize=(10, 6), animation_duration = 5, fps = 20, save_animation = False, 
    potential_saturation=0.8, wavefunction_saturation=0.8, title = "1D interactive bosons"):
        
        total_frames = int(fps * animation_duration)
        dt = js["total time"] / total_frames
       

        fig = plt.figure(figsize=figsize, facecolor='#002b36')
        
        grid = plt.GridSpec(10, 10, hspace=0.4, wspace=6.0)
        ax1 = fig.add_subplot(grid[0:10, 0:5])
        ax2 = fig.add_subplot(grid[3:7, 5:10], sharex=ax1) # probability density of finding any particle at x 
        index = 0
        
        L = js["extent"] / Å
        
        potential_plot = ax1.imshow((Vgrid + Vmin)/(Vmax-Vmin), 
        vmax = 1.0/potential_saturation, vmin = 0, cmap = newcmp, origin = "lower", interpolation = "bilinear", extent = [-L/2, L/2, -L/2, L/2])  
        
        wavefunction_plot = ax1.imshow(complex_to_rgba(Ψ_plot[0], 
        max_val= wavefunction_saturation), origin = "lower", interpolation = "bilinear", extent=[-L / 2,L / 2,-L / 2,L / 2])


        ax1.set_xlabel("$x_1$ [Å]")
        ax1.set_ylabel("$x_2$ [Å]")
        ax1.set_title("$\psi(x_1,x_2)$"+" "+title, color = "white")

        ax2.set_xlabel("$x$ [Å]")
        ax2.set_ylabel("${\| \Psi(x)\|}^{2} $")
        ax2.set_title("Probability density", color = "white")
        
        
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.tick_params(colors='white')
        ax1.spines['left'].set_color('white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['right'].set_color('white') 
        
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['top'].set_linewidth(1)
        ax1.spines['right'].set_linewidth(1)

        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.tick_params(colors='white')
        ax2.spines['left'].set_color('white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white') 
        
        ax2.spines['left'].set_linewidth(1)
        ax2.spines['bottom'].set_linewidth(1)
        ax2.spines['top'].set_linewidth(1)
        ax2.spines['right'].set_linewidth(1)
        

        time_ax = ax2.text(0.97,0.97, "",  color = "white",
                        transform=ax2.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (0.00/femtoseconds)))

       

        if xlim != None:
            ax1.set_xlim(np.array(xlim)/Å)
            ax1.set_ylim(np.array(xlim)/Å)
            ax2.set_xlim(np.array(xlim)/Å)

        prob_density = np.abs(np.sum(  (Ψ_plot[index])*np.conjugate(Ψ_plot[index])  , axis = 1))

        x = np.linspace(-L/2, L/2, js["N"])
        #prob_plot, = ax2.plot(x,  prob_density, color= "cyan")
        #prob_plot_fill = ax2.fill_between(x,prob_density, alpha=0.1, color= "cyan" )
        #ax1.set_aspect('equal', adjustable='box')

        ax2.set_ylim([0,np.amax(prob_density)*1.3])


        #print(total_frames)
        animation_data = {'t': 0.0, 'ax1':ax1 , 'ax2':ax2 ,'frame' : 0, 'max_prob_density' : max(prob_density)*1.3}
        def func_animation(*arg):
            
            time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (animation_data['t']/femtoseconds)))

            animation_data['t'] = animation_data['t'] + dt
            if animation_data['t'] > js["total time"]:
                animation_data['t'] = 0.0

            #print(animation_data['frame'])
            animation_data['frame'] +=1
            index = int((store_steps)/ js["total time"] * animation_data['t'])

            wavefunction_plot.set_data(complex_to_rgba(Ψ_plot[index], max_val= wavefunction_saturation))
            
            if save_animation == True: #avoid reseting the axis makes it faster
                ax2.clear()
                ax2.set_xlabel("$x$ [Å]")
                ax2.set_ylabel("${\| \Psi(x)\|}^{2} $")
                ax2.set_title("Probability density", color = "white")
                ax1.set_facecolor('#002b36')
                ax2.set_facecolor('#002b36')

            prob_density = np.abs(np.sum(  (Ψ_plot[index])*np.conjugate(Ψ_plot[index])  , axis = 1))

            prob_plot, = ax2.plot(x,  prob_density, color= "cyan")
            prob_plot_fill = ax2.fill_between(x,prob_density, alpha=0.1, color= "cyan" )
            new_prob_density = max(prob_density)*1.3
            if new_prob_density > animation_data['max_prob_density']:
                animation_data['max_prob_density'] = new_prob_density

            ax2.set_ylim([0,animation_data['max_prob_density']])

            ax1.set_facecolor('#002b36')
            ax2.set_facecolor('#002b36')

            return potential_plot,wavefunction_plot, time_ax, prob_plot, prob_plot_fill


        frame = 0
        a = animation.FuncAnimation(fig, func_animation,
                                    blit=True, frames=total_frames, interval= 1/fps * 1000)
        if save_animation == True:
            #Writer = animation.writers['ffmpeg']
            #writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            #a.save('animation.mp4', writer=writer)
            if(title == ''):
                title = "animation"
            a.save(js["path save"] + title +'.gif', fps = fps, metadata = dict(artist = 'Me'))
            
        else:
            plt.show()



plot(t = 0, xlim=[-10* Å,10* Å], potential_saturation = 0.5, wavefunction_saturation = 0.2, title = js["title"])


animate(xlim=[-10* Å,10* Å], potential_saturation = 500, wavefunction_saturation = 0.2, 
animation_duration = js["animation duration"], 
save_animation = js["save animation"], title = js["title"])