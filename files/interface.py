from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from model_testing import *

"""Deprecated. This file was used to make a UI for the program, we abandoned this :)"""

class Interface:

    def __init__(self, root):
        # Initialisations
        root.title("Model Interface")
        self.mainframe = ttk.Frame(root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Model frame
        self.model_frame = ttk.Frame(self.mainframe, padding="3 3 12 12")
        self.model_frame.grid(row=0, sticky=(N, W, E, S))

        # TODO: Change this into settings
        self.state = qt.ket2dm(nme_state(np.pi/16))
        config.LHV_type = "vector"

        self.model_address = StringVar()
        self.model_address.set(
            "symmetry\\pi-16_200_SV_singlet\\pi_16_model.h5")
        self.distr = None

        # Load model
        ttk.Label(self.model_frame, text="Model").grid(
            column=0, row=0, sticky=(W, E))
        ttk.Entry(self.model_frame, textvariable=self.model_address, width=50).grid(
            column=1, row=0, sticky=(W, E), columnspan=4)
        ttk.Button(self.model_frame, text="Load", command=self.load).grid(
            column=5, row=0, sticky=(W, E))

        # LHV frame
        self.lhv_frame = ttk.Frame(self.mainframe, padding="3 3 12 12")
        self.lhv_frame.grid(row=1, sticky=(N, W, E, S))

        self.lhv_type_frame = ttk.Frame(self.lhv_frame, padding="3 3 12 12")
        self.lhv_type_frame.grid(row=0, column=0, sticky=(N, W, E, S))

        self.lhv_input_frame = ttk.Frame(self.lhv_frame, padding="3 3 12 12")
        self.lhv_input_frame.grid(row=0, column=1, sticky=(N, W, E, S))

        # LHV settings
        ttk.Label(self.lhv_type_frame, text="LHV Type").grid(
            column=0, row=0, sticky=(W, E))
        self.lhv_type = StringVar()
        self.lhv_type.set('single vector')
        one_vector = ttk.Radiobutton(
            self.lhv_type_frame, text='single 3D vector', variable=self.lhv_type,
            value='single vector', command=self.update_lhv_type)
        semicircle = ttk.Radiobutton(
            self.lhv_type_frame, text='semicircle vector', variable=self.lhv_type,
            value='semicircle', command=self.update_lhv_type)
        one_vector.grid(column=1, row=0)
        semicircle.grid(column=1, row=1)

        self.update_lhv_type()

        # Plot settings frame
        self.plot_settings_frame = ttk.Frame(
            self.mainframe, padding="3 3 12 12")
        self.plot_settings_frame.grid(row=2, sticky=(N, W, E, S))
        ttk.Label(self.plot_settings_frame, text="Settings").grid(
            column=0, row=0, columnspan=3, sticky=(W, E))

        # Plot who
        ttk.Label(self.plot_settings_frame, text='Target plot').grid(
            column=0, row=1, sticky=(W, E))
        self.target = StringVar()
        self.target.set('comm')
        target_box = ttk.Combobox(
            self.plot_settings_frame, textvariable=self.target)
        target_box.state(['readonly'])
        target_box.grid(column=0, row=2)
        target_box['values'] = (
            'comm', 'alice_1', 'alice_2', 'bob_1', 'bob_2')

        # Plot type
        ttk.Label(self.plot_settings_frame, text='Plot type').grid(
            column=1, row=1, sticky=(W, E))
        self.plot_type = StringVar()
        self.plot_type.set('3d')
        target_box = ttk.Combobox(
            self.plot_settings_frame, textvariable=self.plot_type)
        target_box.state(['readonly'])
        target_box.grid(column=1, row=2)
        target_box['values'] = (
            '3d', 'spherical')

        # Buttons
        ttk.Button(self.plot_settings_frame, text="Calculate", command=self.calculate_distr).grid(
            column=2, row=1, sticky=W)
        ttk.Button(self.plot_settings_frame, text="Plot", command=self.plot).grid(
            column=2, row=2, sticky=W)

        # Plotframe
        self.plot_frame = ttk.Frame(self.mainframe, padding="3 3 12 12")
        self.plot_frame.grid(row=3, sticky=(N, W, E, S))

        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
            for grandchild in child.winfo_children():
                grandchild.grid_configure(padx=5, pady=5)

        root.bind("<Return>", self.plot)

    def update_lhv_type(self, *args):
        self.lhv_input_frame.grid_forget()
        self.lhv_input_frame.destroy()
        self.lhv_input_frame = ttk.Frame(self.lhv_frame, padding="3 3 12 12")
        self.lhv_input_frame.grid(row=0, column=1, sticky=(N, W, E, S))
        if self.lhv_type.get() == "single vector":
            # LHV labels
            ttk.Label(self.lhv_input_frame, text="x").grid(
                column=0, row=1, sticky=E)
            ttk.Label(self.lhv_input_frame, text="y").grid(
                column=0, row=2, sticky=E)
            ttk.Label(self.lhv_input_frame, text="z").grid(
                column=0, row=3, sticky=E)
            ttk.Label(self.lhv_input_frame, text="theta").grid(
                column=2, row=1, sticky=E)
            ttk.Label(self.lhv_input_frame, text="phi").grid(
                column=2, row=2, sticky=E)

            # LHV inputs (cartesian)
            self.x = DoubleVar()
            self.y = DoubleVar()
            self.z = DoubleVar()
            x_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.x)
            x_entry.grid(column=1, row=1, sticky=(W, E))
            y_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.y)
            y_entry.grid(column=1, row=2, sticky=(W, E))
            z_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.z)
            z_entry.grid(column=1, row=3, sticky=(W, E))

            # LHV inputs (spherical)
            self.theta = DoubleVar()
            self.phi = DoubleVar()
            theta_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.theta)
            theta_entry.grid(column=3, row=1, sticky=(W, E))
            phi_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.phi)
            phi_entry.grid(column=3, row=2, sticky=(W, E))

            # Spherical or cartesian
            ttk.Label(self.lhv_input_frame, text='Data type').grid(
                column=4, row=1, sticky=(W, E))
            self.data_type = StringVar()
            self.data_type.set('cartesian')
            cartesian = ttk.Radiobutton(
                self.lhv_input_frame, text='cartesian', variable=self.data_type, value='cartesian')
            spherical = ttk.Radiobutton(
                self.lhv_input_frame, text='spherical', variable=self.data_type, value='spherical')
            cartesian.grid(column=4, row=2)
            spherical.grid(column=4, row=3)

        if self.lhv_type.get() == "semicircle":
            # LHV labels
            ttk.Label(self.lhv_input_frame, text="x").grid(
                column=0, row=1, sticky=E)
            ttk.Label(self.lhv_input_frame, text="z").grid(
                column=0, row=2, sticky=E)
            ttk.Label(self.lhv_input_frame, text="theta").grid(
                column=2, row=1, sticky=E)

            # LHV inputs (cartesian)
            self.x = DoubleVar()
            self.z = DoubleVar()
            x_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.x)
            x_entry.grid(column=1, row=1, sticky=(W, E))
            z_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.z)
            z_entry.grid(column=1, row=2, sticky=(W, E))

            # LHV inputs (spherical)
            self.theta = DoubleVar()
            theta_entry = ttk.Entry(
                self.lhv_input_frame, width=5, textvariable=self.theta)
            theta_entry.grid(column=3, row=1, sticky=(W, E))

            # Spherical or cartesian
            ttk.Label(self.lhv_input_frame, text='Data type').grid(
                column=4, row=1, sticky=(W, E))
            self.data_type = StringVar()
            self.data_type.set('cartesian')
            cartesian = ttk.Radiobutton(
                self.lhv_input_frame, text='cartesian', variable=self.data_type, value='cartesian')
            spherical = ttk.Radiobutton(
                self.lhv_input_frame, text='spherical', variable=self.data_type, value='spherical')
            cartesian.grid(column=4, row=2)
            spherical.grid(column=4, row=3)

    def calculate_distr(self, *args):
        if self.lhv_type.get() == 'semicircle':
            if self.data_type.get() == 'cartesian':
                self.normalize()
                vec = [self.x.get(), self.z.get()]
            if self.data_type.get() == 'spherical':
                vec = [np.sin(self.theta.get()),
                       np.cos(self.theta.get())]
        elif self.lhv_type.get() == 'single vector':
            if self.data_type.get() == 'cartesian':
                self.normalize()
                vec = [self.x.get(), self.y.get(), self.z.get()]
            if self.data_type.get() == 'spherical':
                vec = [np.cos(self.phi.get()) * np.sin(self.theta.get()),
                    np.sin(self.phi.get()) * np.sin(self.theta.get()),
                    np.cos(self.theta.get())]
        self.distr = map_distr(self.model, vec, type=self.lhv_type.get())

    def load(self, *args):
        self.model = keras.models.load_model(
            self.model_address.get(), compile=False)

    def plot(self, *args):
        self.calculate_distr()
        cdata = self.distr.c
        adata_1 = self.distr['p_1(a=+1)']
        adata_2 = self.distr['p_2(a=+1)']
        bdata_1 = self.distr['p_1(b=+1)']
        bdata_2 = self.distr['p_2(b=+1)']
        if self.data_type.get() == 'cartesian':
            self.normalize()
            vec = [self.x.get(), self.z.get()]
        if self.data_type.get() == 'spherical':
            vec = [np.sin(self.theta.get()),
                   np.cos(self.theta.get())]

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
            ax.plot([0, 1.25*vec[0]], [0, 0],
                    [0, 1.25*vec[1]], 'r-o', lw=2)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
            fig.colorbar(img, cax=cbar_ax)
        elif self.plot_type.get() == 'spherical':
            theta_data = np.arccos(zdata)
            phi_data = np.arctan2(ydata, xdata)
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            img = ax.scatter(phi_data, theta_data, c=c, vmin=0, vmax=1)
            ax.scatter(np.arctan2(0, vec[0]),
                       np.arccos(vec[1]), c='r', s=20)
            ax.set_xlabel('phi')
            ax.set_ylabel('theta')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
            fig.colorbar(img, cax=cbar_ax)

        # Plotframe
        self.plot_frame = ttk.Frame(self.mainframe, padding="3 3 12 12")
        self.plot_frame.grid(row=3, sticky=(N, W, E, S))

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        if self.plot_type.get() == '3d':
            canvas.mpl_connect('button_press_event', ax._button_press)
            canvas.mpl_connect('button_release_event', ax._button_release)
            canvas.mpl_connect('motion_notify_event', ax._on_move)

    def normalize(self, *args):
        if self.lhv_type.get() == 'semicircle':
            x = self.x.get()
            z = self.z.get()
            norm = np.sqrt(x**2 + z**2)
            self.x.set(abs(x/norm))
            self.z.set(z/norm)
        elif self.lhv_type.get() == 'single vector':
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
