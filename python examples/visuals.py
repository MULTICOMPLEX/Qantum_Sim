from matplotlib import widgets
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from constants import *
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb
import pyfftw
from functions import *


px = 1 / plt.rcParams['figure.dpi']

viridis = cm.get_cmap('gray', 256)
newcolors = viridis(np.linspace(0, 1, 256))
mc = np.array([0, 43/256, 54/256, 1])

newcolors[:150, :] = mc
newcmp = ListedColormap(newcolors)
      
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

def complex_to_rgba(Z: np.ndarray, max_val: float = 1.0):
    r = np.abs(Z)
    arg = np.angle(Z)
    
    h = (arg + np.pi)  / (2 * np.pi)
    s = np.ones(h.shape)
    v = np.ones(h.shape)  #alpha
    rgb = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1)  ) # --> tuple

    abs_z = np.abs(Z)/ max_val
    abs_z = np.where(abs_z> 1., 1. ,abs_z)
    return np.concatenate((rgb, abs_z.reshape((*abs_z.shape,1))), axis= (abs_z.ndim))
    
def complex_plot_2D(
    Ψ_plot = np.ndarray, 
    extent = 10., 
    V = np.ndarray, 
    Vmin = -10., 
    Vmax = 10., 
    t=0, 
    xlim=None, 
    ylim=None, 
    figsize=(640*px, 640*px), 
    potential_saturation=0.5, 
    wavefunction_saturation=0.2
):
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
        
        ax.set_xlabel("[Å]")
        ax.set_ylabel("[Å]")
        ax.set_title("$\psi(x,y,t)$", color='white')

        time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top")
        time_ax.set_text(u"t = {} femtoseconds".format("%.3f"  % (t/const["femtoseconds"])))

        if xlim != None:
            ax.set_xlim(np.array(xlim)/const["femtoseconds"])
        if ylim != None:
            ax.set_ylim(np.array(ylim)/const["femtoseconds"])
        
        
        L = extent/const["femtoseconds"]
 
    
        ax.imshow((V + Vmin)/(Vmax-Vmin), 
        vmax = 1.0/potential_saturation, vmin = 0, cmap = newcmp, origin = "lower", 
        interpolation = "gaussian", extent = [-L/2, L/2, -L/2, L/2])  
      
        
        ax.imshow(complex_to_rgba(Ψ_plot, max_val= wavefunction_saturation), origin = "lower", 
        interpolation = "gaussian", extent = [-L/2, L/2, -L/2, L/2])  
        
        plt.show()
   
def complex_plot(x, phi):
    plt.plot(x, np.abs(phi), label='$|\psi(x)|$')
    plt.plot(x, np.real(phi), label='$Re|\psi(x)|$')
    plt.plot(x, np.imag(phi), label='$Im|\psi(x)|$')
    plt.legend(loc='lower left')
    plt.show()
    return

class ComplexSliderWidget(widgets.AxesWidget):
    """
    A circular complex slider widget for manipulating complex
    values.

    References:
    - https://matplotlib.org/stable/api/widgets_api.
    - https://github.com/matplotlib/matplotlib/blob/
    1ba3ff1c273bf97a65e19892b23715d19c608ae5/lib/matplotlib/widgets.py
    """

    def __init__(self, ax, angle, r, animated=False):
        line, = ax.plot([angle, angle], [0.0, r], linewidth=2.0)
        super().__init__(ax)
        self._rotator = line
        self._is_click = False
        self.animated = animated
        self.update = lambda x, y: None
        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)

    def get_artist(self):
        return self._rotator

    def _click(self, event):
        self._is_click = True
        self._update_plots(event)

    def _release(self, event):
        self._is_click = False

    def on_changed(self, update):
        self.update = update
    
    def _motion(self, event):
        self._update_plots(event)

    def _update_plots(self, event):
        if (self._is_click and event.xdata != None
            and event.ydata != None
            and event.x >= self.ax.bbox.xmin and
            event.x < self.ax.bbox.xmax and
            event.y >= self.ax.bbox.ymin and
            event.y < self.ax.bbox.ymax
            ):
            phi, r = event.xdata, event.ydata 
            if r < 0.2:
                r = 0.0
            self.update(phi, r)
            self._rotator.set_xdata([phi, phi])
            self._rotator.set_ydata([0.0, r])
            if not self.animated:
                event.canvas.draw()

def superpositions(eigenstates, states, energies, extent, fps = 30, total_time = 20, **kw):
        """
        Visualize the time evolution of a superposition of energy eigenstates.
        The circle widgets control the relative phase of each of the eigenstates.
        These widgets are inspired by the circular phasors from the
        quantum mechanics applets by Paul Falstad:
        https://www.falstad.com/qm1d/
        """
        
        total_frames = fps * total_time
   
        coeffs = None
        get_norm_factor = lambda psi: 1.0/(np.sqrt(np.sum(psi*np.conj(psi)))+1e-6)
        animation_data = {'ticks': 0, 'norm': get_norm_factor(eigenstates[0]),
                          'is_paused': False}
        psi0 = eigenstates[0]*get_norm_factor(eigenstates[0])
        if isinstance(states, int) or isinstance(states, float):
            coeffs = np.array([1.0 if i == 0 else 0.0 for i in range(states)],
                           dtype=np.complex128)
            eigenstates = eigenstates[0: states]
        else:
            coeffs = states
            eigenstates = eigenstates[0: len(states)]
            states = len(states)
            psi0 = np.dot(coeffs, eigenstates)
            animation_data['norm'] = get_norm_factor(psi0)
            psi0 *= animation_data['norm']
 
        params = {'dt': 0.001, 
                  'xlim': [-extent/2.0, 
                         extent/2.0],
                  'save_animation': False,
                  'frames': 120
                 }
        for k in kw.keys():
            params[k] = kw[k]

       
        fig = plt.figure(figsize=(16/9 * 5.804 * 0.9, 5.804), facecolor='#002b36') 
        grid = plt.GridSpec(5, states)
        ax = fig.add_subplot(grid[0:3, 0:states])
        
        ax.set_facecolor('#002b36')
               
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        
        ax.set_title("Superpositions", color='white')

        
        ax.set_xlabel("[Å]")
        x = np.linspace(-extent/2.0,
                        extent/2.0,
                        len(eigenstates[0]))
        ax.set_yticks([])
        ax.set_xlim(np.array(params['xlim'])/Å)

        line1, = ax.plot(x/Å, np.real(eigenstates[0]), label='$Re|\psi(x)|$')
        line2, = ax.plot(x/Å, np.imag(eigenstates[0]), label='$Im|\psi(x)|$')
        line3, = ax.plot(x/Å, np.abs(eigenstates[0]), label='$|\psi(x)|$')
        ax.set_ylim(-1.7*np.amax(np.abs(psi0)), 1.7*np.amax(np.abs(psi0)))
        
        leg = ax.legend(facecolor='#002b36', loc='lower left')
        for line, text in zip(leg.get_lines(), leg.get_texts()):
           text.set_color(line.get_color())

        def make_update(n):
            def update(phi, r):
                animation_data['is_paused'] = True
                coeffs[n] = r*np.exp(1.0j*phi)
                psi = np.dot(coeffs, eigenstates)
                animation_data['norm'] = get_norm_factor(psi)
                line1.set_ydata(np.real(psi))
                line2.set_ydata(np.imag(psi))
                line3.set_ydata(np.abs(psi))
            return update

        widgets = []
        circle_artists = []
        for i in range(states):
            circle_ax = fig.add_subplot(grid[4, i], projection='polar')

            plt.setp(circle_ax.spines.values(), color='white')
           
            circle_ax.set_facecolor('#002b36')
            circle_ax.set_title(str(i) # + '\nE=' + str() + '$E_0$'
                                ,color='white')
            circle_ax.set_xticks([])
            circle_ax.set_yticks([])
            
            widgets.append(ComplexSliderWidget(circle_ax, 0.0, 1.0, animated=True))
            widgets[i].on_changed(make_update(i))
            circle_artists.append(widgets[i].get_artist())
        artists = circle_artists + [line1, line2, line3]

        def func(*args):
            animation_data['ticks'] += 1
            e = 1.0
            if animation_data['is_paused']:
                animation_data['is_paused'] = False
            else:
                e *= np.exp(-1.0j*energies[0:states]*params['dt'])
            np.copyto(coeffs, coeffs*e)
            norm_factor = animation_data['norm']
            psi = np.dot(coeffs*norm_factor, eigenstates)
            line1.set_ydata(np.real(psi))
            line2.set_ydata(np.imag(psi))
            line3.set_ydata(np.abs(psi))
            if animation_data['ticks'] % 2:
                return [line1, line2, line3]
            else:
                for i, c in enumerate(coeffs):
                    phi, r = np.angle(c), np.abs(c)
                    artists[i].set_xdata([phi, phi])
                    artists[i].set_ydata([0.0, r])
                return artists
        ani = animation.FuncAnimation(fig, func, blit=True, interval=1000.0/60.0,
                                    frames=None if (not params['save_animation']) else
                                    total_frames)
        if params['save_animation'] == True:
            ani.save('superpositions.gif', fps=fps, metadata=dict(artist='Me'))
            return
        else:
         plt.show()

def animate(
Ψ_plot, 
energies = np.ndarray, 
extent = 10, 
V = np.ndarray, 
Vmin = -10, 
Vmax = 10, 
xlim=np.ndarray, 
ylim=np.ndarray, 
figsize=(7, 7), 
animation_duration = 5, 
fps = 20, 
save_animation = False,
potential_saturation=0.8, 
title = "double slit experiment", 
path_save = "", 
total_time = 10, 
store_steps = 20, 
wavefunction_saturation=0.8):
        
        total_frames = int(fps * animation_duration)
            
        figsize = (640*px, 640*px)

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
                

        L = extent / const["Å"] / 2
        potential_plot = ax.imshow((V + Vmin)/(Vmax-Vmin), 
        vmax = 1.0/potential_saturation, vmin = 0, cmap = newcmp, origin = "lower", 
        interpolation = "gaussian", extent = [-L/2, L/2, -L/2, L/2])
        

        wavefunction_plot = ax.imshow(complex_to_rgba(Ψ_plot[0], max_val= wavefunction_saturation),
        origin = "lower", interpolation = "gaussian", extent=[-L/2, L/2, -L/2, L/2])


        if xlim != None:
            ax.set_xlim(np.array(xlim)/const["Å"])
        if ylim != None:
            ax.set_ylim(np.array(ylim)/const["Å"])

        

        ax.set_title("$\psi(x,y,t)$"+" "+title, color = "white")
        ax.set_xlabel('[Å]')
        ax.set_ylabel('[Å]')

        time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top", alpha=0.9)
        
        energy_ax = ax.text(0.97,0.93, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top", alpha=0.9)
        

        xdt = np.linspace(0, total_time/const["femtoseconds"], total_frames)
        psi_index = np.linspace(0, store_steps, total_frames)
        
        def func_animation(frame):
            
            time_ax.set_text(u"time = {} femtoseconds".format("%.3f" % (xdt[frame])))
            
            index = int(psi_index[frame])
            wavefunction_plot.set_data(complex_to_rgba(Ψ_plot[index], max_val= wavefunction_saturation))
            
            formatted_num = "{:14.14e}".format(np.abs(energies[index]))
            energy_ax.set_text(u"Energy =  "+ formatted_num)
            
            return wavefunction_plot, time_ax


        ani = animation.FuncAnimation(fig, func_animation,
                                    blit=True, frames=total_frames, interval= 1/fps * 1000)
        if save_animation == True:
            if(title == ''):
                title = "animation"
            ani.save(path_save + title +'.gif', fps = fps, metadata = dict(artist = 'Me'))
        else:
            plt.show()
            
        
        