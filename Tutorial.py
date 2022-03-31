#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic('matplotlib', 'tk')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.text import Text
import tkinter as tk
import numpy as np
import sys

# Constants
mass  = 1000        # 1,000 kg
area  = 7.5         # 7.5 m^2
F_max = 10000       # 10,000 N
mps_to_mph  = 2.237 # 1 m/s = ~2.237 miles/hr
mps_to_kmph = 3.6   # 1 m/2 = 3.6 km/hr
g = 9.80665         # Gravitational acceleration in m/s^2
interval = 1000     # 1,000 ms = 1 s
y_max    = 140      # 140 mph
setpoint = 29.0576  # 29.0576 m/s = 65 mph

class Controller:
    def get_output(self, _):
        return 0.0

    def set_gain(self, _):
        pass

    def set_kp(self, _):
        pass

    def set_ki(self, _):
        pass

    def set_kd(self, _):
        pass

    def get_pid_outputs(self):
        return 0.0, 0.0, 0.0

class FeedForwardController(Controller):
    def __init__(self):
        self.gain = 0

    def set_gain(self, gain):
        self.gain = gain

    def get_output(self, _):
        return self.gain

class PIDController(Controller):
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.cum_error = 0.0
        self.e_prev = 0.0
        self.up, self.ui, self.ud = 0, 0, 0
        
    def get_output(self, e):
        self.cum_error += e
        d = e - self.e_prev
        self.e_prev = e
        self.up = self.kp * e
        self.ui = self.ki * self.cum_error
        self.ud = self.kd * d
        output = self.up + self.ui + self.ud
        return np.clip(output, 0, 1.0)

    def set_kp(self, k):
        self.kp = k

    def set_ki(self, k):
        self.ki = k

    def set_kd(self, k):
        self.kd = k

    def get_pid_outputs(self):
        return self.up, self.ui, self.ud

class System:
    def __init__(self, options):
        self.v    = 0      # initial velocity, 0 m/s
        self.d    = 0      # initial distance, 0 m
        self.elev = 0      # initial elevation, 0m
        self.e    = 0.0    # initial error, 0.0 m/s
        self.controller = None
        self.options = options
        self.terrain = self.init_terrain()

    def set_controller(self, controller):
        self.controller = controller
    
    def init_terrain(self):
        slope = np.repeat([0, 0.5, 0, -0.25, 0, 0.25, -0.5], [5, 5, 5, 10, 5, 10, 5])
        return np.tile(slope, 10)

    def get_controller_output(self, e):
        return self.controller.get_output(e)

    def get_slope(self, d):
        index = int(d/100) % len(self.terrain)
        return self.terrain[index]

    def calc_F_terrain(self, d):
        return np.sin(np.arctan(self.get_slope(self.d))) * g * mass  # Backward force due to gravity

    def update_system(self, t, system_input):
        F_engine  = system_input * F_max         # Forward force due to engine
        F_drag    = 0.5 * self.v**2 * area       # Backward force due to aero drag
        if self.options.elevation:
            F_terrain = self.calc_F_terrain(self.d)  # Backward force due to slope of terrain
        else:
            F_terrain = 0.0
        self.v += (F_engine - F_drag - F_terrain)/mass
        self.d += self.v
        self.elev += self.v * self.get_slope(self.d)
        return self.v

class ScrollingGraph:
    def __init__(self, ax, xlabel, xlim, ylabel, ylim, target=None):
        self.ax     = ax
        self.xdata  = []
        self.ydata  = []
        self.line,  = ax.plot(self.xdata, self.ydata, lw=2)
        self.target = target
        ax.grid()
        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        
        if target is not None:
            ax.axhline(y=target, color="red", linestyle="--")
        
    def append(self, x, y):
        self.xdata.append(x)
        self.ydata.append(y)
        self.check_bounds(x, y)
        self.line.set_data(self.xdata, self.ydata)
        return self.line
        
    def check_bounds(self, x, y):
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        
        if x > xmax:
            self.ax.set_xlim(xmin+(x-xmax), x)
            self.ax.figure.canvas.draw_idle()
        
        if y < ymin:
            self.ax.set_ylim(y*1.1, ymax)
            self.ax.figure.canvas.draw_idle()

        if y > ymax:
            self.ax.set_ylim(ymin, y*1.1)
            self.ax.figure.canvas.draw_idle()


class Display:
    def __init__(self, system, options):
        self.root   = None
        self.fig    = None
        self.canvas = None
        self.velocity_graph  = None
        self.error_graph     = None
        self.elevation_graph = None
        self.up_graph = None
        self.ui_graph = None
        self.ud_graph = None
        self.gain_slider = None
        self.ani = None
        self.system = system
        self.options = options
        self.started = False
        
        self.root = tk.Tk()
        self.root.wm_title("Car Simulation")

        # label = tk.Label(root,text="Car Simulation").grid(column=0, row=0)

        self.fig = plt.Figure(figsize=(12,10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(column=0,row=0)

        gs_kw = dict(height_ratios=[2, 2, 2, 1, 0.5, 0.5])
        self.axd = axd = self.fig.subplot_mosaic([['velocity', 'velocity', 'velocity'],
                                                  ['error', 'error', 'error'],
                                                  ['elevation', 'elevation', 'elevation'],
                                                  ['up', 'ui', 'ud'],
                                                  ['Kp', 'Ki', 'Kd'],
                                                  ['start', 'gain', 'gain']],
                                                  gridspec_kw=gs_kw)

        # self.axd = axd = self.fig.subplot_mosaic([['velocity'],
        #                                          ['elevation'],
        #                                          ['gain']])

        self.velocity_graph = ScrollingGraph(axd['velocity'], 'Time [s]', (0, 20),
                                            'Velocity [mi/hr]', (-10, y_max),
                                            target=setpoint*mps_to_mph)

        self.error_graph = ScrollingGraph(axd['error'], 'Time [s]', (0, 20),
                                             'Error', (-20, 80), target=0.0)

        self.elevation_graph = ScrollingGraph(axd['elevation'], 'Time [s]', (0, 20),
                                             'Elevation [m]', (-10.0, 100))

        self.up_graph = ScrollingGraph(axd['up'], 'Time [s]', (0, 20), 'up', (-3.0, 3))
        self.ui_graph = ScrollingGraph(axd['ui'], 'Time [s]', (0, 20), 'ui', (-3.0, 3))
        self.ud_graph = ScrollingGraph(axd['ud'], 'Time [s]', (0, 20), 'ud', (-3.0, 3))

        self.kp_text = TextBox(axd['Kp'], "Kp")
        self.kp_text.on_submit(self.kp_submit)
        self.kp_text.set_val("0.0")

        self.ki_text = TextBox(axd['Ki'], "Ki")
        self.ki_text.on_submit(self.ki_submit)
        self.ki_text.set_val("0.0")

        self.kd_text = TextBox(axd['Kd'], "Kd")
        self.kd_text.on_submit(self.kd_submit)
        self.kd_text.set_val("0.0")

        # adjust the main plot to make room for the slider
        # fig.subplots_adjust(bottom=0.20, hspace=0.4)

        # Make a horizontal slider to control the accelerator
        # ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
        self.gain_slider = Slider(
                ax=axd['gain'],
                label='Accelerator',
                valmin=0.0,
                valmax=1.0,
                valinit=0
        )
        self.gain_slider.on_changed(self.set_gain)

        self.start_button = Button(axd['start'], 'Start', hovercolor='green')
        self.start_button.on_clicked(self.start_animation)

        axd['elevation'].set_visible(False)
        axd['Kp'].set_visible(False)
        axd['Ki'].set_visible(False)
        axd['Kd'].set_visible(False)
        axd['gain'].set_visible(False)

        if self.options.elevation:
            axd['elevation'].set_visible(True)

        if self.options.kp:
            axd['Kp'].set_visible(True)

        if self.options.ki:
            axd['Ki'].set_visible(True)

        if self.options.kd:
            axd['Kd'].set_visible(True)

        if self.options.gain:
            axd['gain'].set_visible(True)

        self.fig.tight_layout()
    
    def data_gen(self, t=0):
        e = 0
        v = 0
        while True:
            t += 1
            e = setpoint - v
            system_input = self.system.get_controller_output(e)
            up, ui, ud = self.system.controller.get_pid_outputs()
            v = self.system.update_system(t, system_input)
            yield t, v*mps_to_mph, self.system.elev, setpoint-v, up, ui, ud

    def init(self):
        return self.velocity_graph.append(0, 0), self.error_graph.append(0, 0), self.elevation_graph.append(0, 0), \
               self.up_graph.append(0, 0), self.ui_graph.append(0, 0), self.ud_graph.append(0, 0)

    def run(self, data):
        t, v, l, e, up, ui, ud = data
        return self.velocity_graph.append(t, v), \
               self.error_graph.append(t, e), \
               self.elevation_graph.append(t, l), \
               self.up_graph.append(t, np.clip(up, -3, 3)), \
               self.ui_graph.append(t, np.clip(ui, -3, 3)), \
               self.ud_graph.append(t, np.clip(ud, -3, 3))


    def kp_submit(self, text):
        self.system.controller.set_kp(float(text))

    def ki_submit(self, text):
        self.system.controller.set_ki(float(text))

    def kd_submit(self, text):
        self.system.controller.set_kd(float(text))

    def set_gain(self, val):
        if self.options.one_shot and not self.started:
            self.system.controller.set_gain(val)
        elif self.options.gain and not self.options.one_shot:
            self.system.controller.set_gain(val)
        self.fig.canvas.draw_idle()

    def start_animation(self, event):
        if not self.started:
            self.ani = animation.FuncAnimation(self.fig, self.run, self.data_gen, blit=False,
                                               interval=interval, repeat=False, init_func=self.init)
            self.started = True

    def show(self):
        plt.show()
        self.root.mainloop()

class Options:
    elevation = False
    kp = False
    ki = False
    kd = False
    one_shot = False
    gain = False

def main(args):
    if (len(args)<2):
        print("Usage: {} mode".format(args[0]))
        sys.exit(1)

    if args[1] not in ['feedforward', 'feedback', 'uncertainty', 'p_control', 'pi_control', 'pid_control']:
        print("Invalid argument")
        sys.exit(1)

    mode = args[1]
    options = Options()

    if mode=='feedforward':
        options.one_shot = True
        options.gain = True
    elif mode=='feedback':
        options.gain = True
    elif mode=='uncertainty':
        options.gain = True
        options.elevation = True
    elif mode=='p_control':
        options.elevation = True
        options.kp = True
    elif mode=='pi_control':
        options.elevation = True
        options.kp = True
        options.ki = True
    elif mode=='pid_control':
        options.elevation = True
        options.kp = True
        options.ki = True
        options.kd = True
    else:
        print("Unrecognized mode: {}".format((args[1])))

    if options.gain:
        controller = FeedForwardController()
    else:
        controller = PIDController()

    system = System(options)
    system.set_controller(controller)
    display = Display(system, options)
    display.show()

if __name__ == "__main__":
    main(sys.argv)
