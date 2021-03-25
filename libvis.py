import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class Visualizations:
    @staticmethod
    def get_3d_vis_data(sim, resolution=1):
        x = []
        y = []
        z = []
        values = []
        sample = 1
        for point, value in np.ndenumerate(sim.V):
            if sample >= resolution:
                unit_pt = sim.point_to_global_unit(point)
                x.append(unit_pt[0])
                y.append(unit_pt[1])
                z.append(unit_pt[2])
                values.append(value)
                sample = 1
            else:
                sample += 1
        return x, y, z, values
    
    @staticmethod
    def colormesh_3d(sim, size=(10, 10), color_norm="auto", resolution="auto"):
        fig = plt.figure(figsize=size)
        ax = plt.axes(projection='3d')

        if resolution == "auto":
            resolution = max(int(np.average(size) / np.average(sim.space_size)), 1)
            print("Set visualization resolution to", resolution)
        x, y, z, values = Visualizations.get_3d_vis_data(sim, resolution)

        cmap = plt.cm.RdBu
        custom_cmap = cmap(np.arange(cmap.N))
        custom_cmap[:,-1] = np.concatenate((np.linspace(1, 0, cmap.N // 2), np.linspace(0, 1, cmap.N // 2)))
        custom_cmap = ListedColormap(custom_cmap)

        if color_norm != None:
            if color_norm == "auto":
                flat_V = sim.V.flatten()
                color_norm = abs(max(abs(max(flat_V)), abs(min(flat_V))))
            ax.scatter(x, y, z, c=values, marker="p", cmap=custom_cmap, vmin=-color_norm, vmax=color_norm)
        else:
            ax.scatter(x, y, z, c=values, marker="p", cmap=custom_cmap)
        ax.set_xlabel(sim.axis_names[0])
        ax.set_ylabel(sim.axis_names[1])
        ax.set_zlabel(sim.axis_names[2])
        plt.show()
        
    @staticmethod
    def colormesh_2d(sim, size=(10, 10), color_norm="auto"):
        fig = plt.figure(figsize=size)
        ax = fig.gca()
        x, y = sim.get_meshgrid()
        
        if color_norm != None:
            if color_norm == "auto":
                flat_V = sim.V.flatten()
                color_norm = abs(max(abs(max(flat_V)), abs(min(flat_V))))
            ax.pcolormesh(x, y, sim.V.T, cmap="RdBu", shading="auto", vmin=-color_norm, vmax=color_norm)
        else:
            ax.pcolormesh(x, y, sim.V.T, cmap="RdBu", shading="auto")
        ax.set_xlabel(sim.axis_names[0])
        ax.set_ylabel(sim.axis_names[1])
        plt.show()
        
    @staticmethod
    def contour_2d(sim, size=(10, 10)):
        fig = plt.figure(figsize=size)
        ax = fig.gca()
        x, y = sim.get_meshgrid()
        
        contours = ax.contour(x, y, sim.V.T)
        ax.clabel(contours)
        ax.set_xlabel(sim.axis_names[0])
        ax.set_ylabel(sim.axis_names[1])
        plt.show()

