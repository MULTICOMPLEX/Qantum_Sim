from matplotlib import widgets
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from constants import *


def differentiate_twice(f, p2):
    f = np.fft.ifftn(-p2*np.fft.fftn(f))
    return f

def norm(phi, dx):
    norm = np.linalg.norm(phi) * dx
    return (phi * np.sqrt(dx)) / norm

def norm2(phi, dx):
    norm = np.sum(np.square(np.abs(phi)))*dx
    return phi/np.sqrt(norm)

def apply_projection(tmp, psi_list, dx):
    for psi in psi_list:
        tmp -= np.vdot(psi, tmp) * psi * dx
    return tmp

def apply_projection2(tmp, psi_list, dx):
    for psi in psi_list:
        tmp -= np.sum(tmp*np.conj(psi)) * psi * dx
    return tmp

def Split_Step_NP(Ψ, phi, dx, store_steps, Nt_per_store_step, Ur, Uk, ite):
    for i in range(store_steps):
        tmp = Ψ[i]
        for _ in range(Nt_per_store_step):
            c = np.fft.fftn(Ur*tmp)
            tmp = Ur * np.fft.ifftn(Uk*c)
            if(ite):
              tmp = apply_projection(tmp, phi, dx)
        if(ite):
           Ψ[i+1] = norm(tmp, dx)
        else:
           Ψ[i+1] = tmp 
    return
    
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
        
        