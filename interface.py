from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from model_testing import *


class Interface:

    def __init__(self, root):

        folder_name = "new-LHV\\pi-16_200_SV\\"
        self.state = qt.ket2dm(nme_state(np.pi/16))
        config.LHV_type = "vector"
        self.model = keras.models.load_model(
            folder_name + "pi_16_model.h5", compile=False)
        self.distr = None

        root.title("Model Interface")

        self.width = 7
        self.length = 4

        self.mainframe = ttk.Frame(root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Titles
        ttk.Label(self.mainframe, text="LHV vector").grid(
            column=0, row=0, columnspan=4, sticky=(W, E))
        ttk.Label(self.mainframe, text="Settings").grid(
            column=4, row=0, columnspan=4, sticky=(W, E))

        # LHV labels
        ttk.Label(self.mainframe, text="x").grid(
            column=0, row=1, sticky=E)
        ttk.Label(self.mainframe, text="y").grid(
            column=0, row=2, sticky=E)
        ttk.Label(self.mainframe, text="z").grid(
            column=0, row=3, sticky=E)
        ttk.Label(self.mainframe, text="theta").grid(
            column=2, row=1, sticky=E)
        ttk.Label(self.mainframe, text="phi").grid(
            column=2, row=2, sticky=E)

        # LHV inputs (cartesian)
        self.x = DoubleVar()
        self.y = DoubleVar()
        self.z = DoubleVar()
        x_entry = ttk.Entry(
            self.mainframe, width=5, textvariable=self.x)
        x_entry.grid(column=1, row=1, sticky=(W, E))
        y_entry = ttk.Entry(
            self.mainframe, width=5, textvariable=self.y)
        y_entry.grid(column=1, row=2, sticky=(W, E))
        z_entry = ttk.Entry(
            self.mainframe, width=5, textvariable=self.z)
        z_entry.grid(column=1, row=3, sticky=(W, E))

        # LHV inputs (spherical)
        self.theta = DoubleVar()
        self.phi = DoubleVar()
        theta_entry = ttk.Entry(
            self.mainframe, width=5, textvariable=self.theta)
        theta_entry.grid(column=3, row=1, sticky=(W, E))
        phi_entry = ttk.Entry(
            self.mainframe, width=5, textvariable=self.phi)
        phi_entry.grid(column=3, row=2, sticky=(W, E))

        # Spherical or cartesian
        ttk.Label(self.mainframe, text='Data type').grid(
            column=4, row=1, sticky=(W, E))
        self.data_type = StringVar()
        self.data_type.set('cartesian')
        cartesian = ttk.Radiobutton(
            self.mainframe, text='cartesian', variable=self.data_type, value='cartesian')
        spherical = ttk.Radiobutton(
            self.mainframe, text='spherical', variable=self.data_type, value='spherical')
        cartesian.grid(column=4, row=2)
        spherical.grid(column=4, row=3)

        # Plot who
        ttk.Label(self.mainframe, text='Target plot').grid(
            column=5, row=1, sticky=(W, E))
        self.target = StringVar()
        self.target.set('comm')
        target_box = ttk.Combobox(self.mainframe, textvariable=self.target)
        target_box.state(['readonly'])
        target_box.grid(column=5, row=2)
        target_box['values'] = (
            'comm', 'alice_1', 'alice_2', 'bob_1', 'bob_2')

        # Plot type
        ttk.Label(self.mainframe, text='Plot type').grid(
            column=6, row=1, sticky=(W, E))
        self.plot_type = StringVar()
        self.plot_type.set('3d')
        target_box = ttk.Combobox(self.mainframe, textvariable=self.plot_type)
        target_box.state(['readonly'])
        target_box.grid(column=6, row=2)
        target_box['values'] = (
            '3d', 'spherical')

        # Buttons
        ttk.Button(self.mainframe, text="Calculate", command=self.calculate_distr).grid(
            column=3, row=3, sticky=W)
        ttk.Button(self.mainframe, text="Plot", command=self.plot).grid(
            column=6, row=3, sticky=W)

        self.plotframe = ttk.Frame(self.mainframe, padding="3 3 12 12")
        self.plotframe.grid(column=0, row=self.length,
                            columnspan=self.width, sticky=(N, W, E, S))

        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        x_entry.focus()
        root.bind("<Return>", self.plot)

    def calculate_distr(self, *args):
        if self.data_type.get() == 'cartesian':
            self.normalize()
            self.theta.set(0)
            self.phi.set(0)
            vec = [self.x.get(), self.y.get(), self.z.get()]
        if self.data_type.get() == 'spherical':
            self.x.set(0)
            self.y.set(0)
            self.z.set(0)
            pass
        self.distr = map_distr_SV(self.model, vec)

    def plot(self, *args):
        cdata = self.distr.c
        adata_1 = self.distr['p_1(a=0)']
        adata_2 = self.distr['p_2(a=0)']
        bdata_1 = self.distr['p_1(b=0)']
        bdata_2 = self.distr['p_2(b=0)']
        
        if self.target.get() == 'comm':
            c = cdata
            axes = 'alice'
        elif self.target.get() == 'alice_1':
            c = adata_1
            axes = 'alice'
        elif self.target.get() == 'alice_2':
            c = adata_2
            axes = 'alice'
        elif self.target.get() == 'bob_1':
            c = bdata_1
            axes = 'bob'
        elif self.target.get() == 'bob_2':
            c = bdata_2
            axes = 'bob'
        
        if axes == 'alice':
            xdata = self.distr.ax
            ydata = self.distr.ay
            zdata = self.distr.az
        elif axes == 'bob':
            xdata = self.distr.bx
            ydata = self.distr.by
            zdata = self.distr.bz
        
        if self.plot_type.get() == '3d':
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(xdata, ydata, zdata, c=c, vmin=0, vmax=1)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
            fig.colorbar(img, cax=cbar_ax)
        elif self.plot_type.get() == 'spherical':
            theta_data = np.arccos(zdata)
            phi_data = np.arctan2(ydata, xdata)
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            img = ax.scatter(phi_data, theta_data, c=c, vmin=0, vmax=1)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
            fig.colorbar(img, cax=cbar_ax)

        self.plotframe = ttk.Frame(self.mainframe, padding="3 3 12 12")
        self.plotframe.grid(column=0, row=self.length,
                            columnspan=self.width, sticky=(N, W, E, S))

        canvas = FigureCanvasTkAgg(fig, master=self.plotframe)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.plotframe)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        if self.plot_type.get() == '3d':
            canvas.mpl_connect('button_press_event', ax._button_press)
            canvas.mpl_connect('button_release_event', ax._button_release)
            canvas.mpl_connect('motion_notify_event', ax._on_move)

    def normalize(self, *args):
        x = self.x.get()
        y = self.y.get()
        z = self.z.get()
        norm = np.sqrt(x**2 + y**2 + z**2)
        self.x.set(x/norm)
        self.y.set(y/norm)
        self.z.set(z/norm)


root = Tk()
Interface(root)
root.mainloop()
