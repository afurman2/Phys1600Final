import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import libem
from numba import jit, float64

class Visualizations:
    @staticmethod
    @jit
    def get_3d_vis_data(V, scale, top_left, resolution=1):
        x = np.zeros(V.size, float64)
        y = np.zeros(V.size, float64)
        z = np.zeros(V.size, float64)
        values = np.zeros(V.size, float64)
        sample = 1
        n = 0
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                for k in range(V.shape[2]):
                    if sample >= resolution:
                        x[n] = top_left[0] + (i / scale)
                        y[n] = top_left[1] + (j / scale)
                        z[n] = top_left[2] + (k / scale)
                        values[n] = V[i,j,k]
                        sample = 1
                    else:
                        sample += 1
                    n += 1
        #for point, value in np.ndenumerate(sim.V):
        #    if sample >= resolution:
        #        unit_pt = sim.point_to_global_unit(point)
        #        x[i] = unit_pt[0]
        #        y[i] = unit_pt[1]
        #        z[i] = unit_pt[2]
        #        values[i] = value
        #        sample = 1
        #    else:
        #        sample += 1
        #    i += 1  
        
        return x, y, z, values
    
    @staticmethod
    def colormesh_3d(sim, size=(10, 10), color_norm="auto", resolution="auto"):
        fig = plt.figure(figsize=size)
        ax = plt.axes(projection='3d')

        if resolution == "auto":
            resolution = max(int(np.average(size) / np.average(sim.space_size)), 1)
            print("Set visualization resolution to", resolution)
        x, y, z, values = Visualizations.get_3d_vis_data(sim.V, sim.scale, sim.top_left, resolution)

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
    def color_xsections_3d(sim3d, ax_loc, size=(10, 10), color_norm="auto", resolution="auto"):        
        graph_V = np.zeros(sim3d.V.shape, float)
        for axis, location in ax_loc:
            sim2d = libem.EMSimulationSpace2D.from_3d(sim3d, axis, location)
            if axis == 0:
                loc = sim3d.global_unit_to_point((location, 0, 0))
                graph_V[loc[0],:,:] = sim2d.V
            elif axis == 1:
                loc = sim3d.global_unit_to_point((0, location, 0))
                graph_V[:,loc[1],:] = sim2d.V
            elif axis == 2:
                loc = sim3d.global_unit_to_point((0, 0, location))
                graph_V[:,:,loc[2]] = sim2d.V
            
        dummy_sim = libem.EMSimulationSpace3D(sim3d.space_size, sim3d.scale, sim3d.top_left, sim3d.axis_names)
        dummy_sim.V = graph_V
        
        Visualizations.colormesh_3d(dummy_sim, size, color_norm, resolution)
        
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
        
        cnt_levels = []
        sampled_V = sim.V.flatten()
        min_sV = min(sampled_V)
        max_sV = max(sampled_V)
        std_sV = np.std(sampled_V)
        steps = max(int(abs(max_sV - min_sV) / std_sV), 8)
        prev_lvl = 0
        for i in range(steps):
            lvl = min_sV + ((abs(max_sV - min_sV) / steps) * i)
            if prev_lvl < 0 and lvl > 0:
                cnt_levels.append(0)
            cnt_levels.append(lvl)
            prev_lvl = lvl
                                
        contours = ax.contour(x, y, sim.V.T, levels=cnt_levels)
        ax.clabel(contours)
        ax.set_xlabel(sim.axis_names[0])
        ax.set_ylabel(sim.axis_names[1])
        plt.show()

