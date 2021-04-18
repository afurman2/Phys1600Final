import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import libem
from numba import jit, float64

import os
from shutil import rmtree

V_COLORS = "RdBu"#"inferno"
T_COLORS = "Greens"

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
        return x, y, z, values
    
    @staticmethod
    @jit
    def get_2d_vis_data(V, scale, top_left, resolution=1):
        x = np.zeros(V.size, float64)
        y = np.zeros(V.size, float64)
        values = np.zeros(V.size, float64)
        sample = 1
        n = 0
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if sample >= resolution:
                    x[n] = top_left[0] + (i / scale)
                    y[n] = top_left[1] + (j / scale)
                    values[n] = V[i,j]
                    sample = 1
                else:
                    sample += 1
                n += 1        
        return x, y, values
    
    @staticmethod
    def colormesh_3d(sim, size=(10, 10), color_norm="auto", resolution="auto", graph_ax=None):
        ax = None
        if graph_ax is None:
            plt.figure(figsize=size)
            ax = plt.axes(projection='3d')
        else:
            ax = graph_ax

        if resolution == "auto":
            resolution = max(int(np.average(size) / np.average(sim.space_size)), 1)
            print("Set visualization resolution to", resolution)
        x, y, z, values = Visualizations.get_3d_vis_data(sim.V, sim.scale, sim.top_left, resolution)

        cmap = plt.cm.RdBu if V_COLORS == "RdBu" else plt.cm.inferno
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
        
        if graph_ax is None:
            ax.set_xlabel(sim.axis_names[0])
            ax.set_ylabel(sim.axis_names[1])
            ax.set_zlabel(sim.axis_names[2])
            plt.show()
        
    @staticmethod
    def color_xsections_3d(sim3d, ax_loc, size=(10, 10), color_norm="auto", resolution="auto", graph_ax=None):
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
        Visualizations.colormesh_3d(dummy_sim, size, color_norm, resolution, graph_ax)
        
    @staticmethod
    def colormesh_2d(sim, size=(10, 10), color_norm="auto", graph_ax=None):
        fig = None
        ax = None
        if graph_ax is None:
            fig = plt.figure(figsize=size)
            ax = fig.gca()
        else:
            ax = graph_ax
        
        x, y = sim.get_meshgrid()
        
        if color_norm != None:
            if color_norm == "auto":
                flat_V = sim.V.flatten()
                color_norm = abs(max(abs(max(flat_V)), abs(min(flat_V))))
            ax.pcolormesh(x, y, sim.V.T, cmap=V_COLORS, shading="auto", vmin=-color_norm, vmax=color_norm)
        else:
            ax.pcolormesh(x, y, sim.V.T, cmap=V_COLORS, shading="auto")
        
        if graph_ax is None:
            ax.set_xlabel(sim.axis_names[0])
            ax.set_ylabel(sim.axis_names[1])
            plt.show()
        
    @staticmethod
    def contour_2d(sim, size=(10, 10), graph_ax=None):
        fig = None
        ax = None
        if graph_ax is None:
            fig = plt.figure(figsize=size)
            ax = fig.gca()
        else:
            ax = graph_ax
            
        x, y = sim.get_meshgrid()
        
        cnt_levels = []
        sampled_V = sim.V.flatten()
        min_sV = min(sampled_V)
        max_sV = max(sampled_V)
        std_sV = np.std(sampled_V)
        steps = max(int(abs(max_sV - min_sV) / std_sV), 8)
        prev_lvl = 0
        for i in range(steps + 1):
            lvl = min_sV + ((abs(max_sV - min_sV) / steps) * i)
            if prev_lvl < 0 and lvl > 0:
                cnt_levels.append(0)
            cnt_levels.append(lvl)
            prev_lvl = lvl
                                
        contours = ax.contour(x, y, sim.V.T, levels=cnt_levels)
        ax.clabel(contours)
        
        if graph_ax is None:
            ax.set_xlabel(sim.axis_names[0])
            ax.set_ylabel(sim.axis_names[1])
            plt.show()

    @staticmethod
    def efield_3d(sim3d, size=(10, 10), resolution="auto", graph_ax=None):
        ax = None
        if graph_ax is None:
            plt.figure(figsize=size)
            ax = plt.axes(projection='3d')
        else:
            ax = graph_ax
            
        if resolution == "auto":
            resolution = max(int((np.average(size) * sim3d.scale) / (3 * np.average(sim3d.space_size))), 10)
            print("Set visualization resolution to", resolution)
        
        E_x, E_y, E_z = sim3d.get_efield()
        
        x, y, z, E_x = Visualizations.get_3d_vis_data(E_x, sim3d.scale, sim3d.top_left, resolution)
        _, _, _, E_y = Visualizations.get_3d_vis_data(E_y, sim3d.scale, sim3d.top_left, resolution)
        _, _, _, E_z = Visualizations.get_3d_vis_data(E_z, sim3d.scale, sim3d.top_left, resolution)        
        
        ax.quiver3D(x, y, z, E_x, E_y, E_z, label="E")
        
        if graph_ax is None:
            ax.set_xlabel(sim3d.axis_names[0])
            ax.set_ylabel(sim3d.axis_names[1])
            ax.set_zlabel(sim3d.axis_names[2])
            plt.show()
        
    @staticmethod
    def efield_2d(sim2d, size=(10, 10), graph_ax=None):
        fig = None
        ax = None
        if graph_ax is None:
            fig = plt.figure(figsize=size)
            ax = fig.gca()
        else:
            ax = graph_ax
            
        E_x, E_y = sim2d.get_efield()
        
        x, y, E_x = Visualizations.get_2d_vis_data(E_x, sim2d.scale, sim2d.top_left, 1)
        _, _, E_y = Visualizations.get_2d_vis_data(E_y, sim2d.scale, sim2d.top_left, 1)
        
        ax.quiver(x, y, E_x, E_y, label="E")
        
        if graph_ax is None:
            ax.set_xlabel(sim2d.axis_names[0])
            ax.set_ylabel(sim2d.axis_names[1])
            plt.show()
            
    @staticmethod
    def trajectory_3d(time, x, size=(10, 10), graph_ax=None):
        ax = None
        if graph_ax is None:
            plt.figure(figsize=size)
            ax = plt.axes(projection='3d')
        else:
            ax = graph_ax
        
        ax.scatter(x[0], x[1], x[2], c=time, cmap=T_COLORS)
        
        if graph_ax is None:
            plt.show()
            
    @staticmethod
    def trajectory_2d(time, x3d, axis=0, size=(10, 10), graph_ax=None):
        ax = None
        if graph_ax is None:
            plt.figure(figsize=size)
            ax = plt.gca()
        else:
            ax = graph_ax
                    
        x = np.delete(x3d, axis, axis=0)
        
        ax.scatter(x[0], x[1], c=time, cmap=T_COLORS)
        
        if graph_ax is None:
            plt.show()
            
class VideoMaker(object):
    def __init__(self, figure, axes, videoDir=None, framerate=1):
        self.fig = figure
        self.axes = np.array(axes)
        self.framerate = framerate
        
        self.curr_frame = -1
        
        self.videoDir = "video_tmp" if videoDir is None else videoDir
        if os.path.exists(self.videoDir):
            rmtree(self.videoDir)
        os.mkdir(self.videoDir)
        
    def new_frame(self):
        self.curr_frame += 1
        for axis in self.axes.flatten():
            axis.clear()
            
    def draw_frame(self, save=True):
        self.fig.canvas.draw()
        if save:
            plt.savefig(os.path.join(self.videoDir, "frame{:03d}.png".format(self.curr_frame)))
        
    def make_movie(self, name="movie.mp4"):
        cwd = os.getcwd()
        os.chdir(self.videoDir)
        os.system("ffmpeg -framerate {} -i frame%03d.png -r 24 -pix_fmt yuv420p {}".format(self.framerate, name))
        os.chdir(cwd)
        