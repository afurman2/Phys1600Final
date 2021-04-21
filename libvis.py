import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import libem

import os
from shutil import rmtree

"""
The Electrodynamics Simulation Library libvis.py
Author: Adam Furman
Brown University

Provides methods of visualizing electrodynamics simulations.
"""

V_COLORS = "RdBu"#"inferno"
T_COLORS = "Greens"

class Visualizations: 
    @staticmethod
    def get_3d_vis_data(V, scale, top_left, resolution=1):
        """
        Pull a subset of points from the array V to use in a 3D visualization.
        """
        x = []
        y = []
        z = []
        values = []
        
        for pt, value in np.ndenumerate(V[::resolution,::resolution,::resolution]):
            loc = ((np.array(pt) * resolution) / scale) + top_left
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])
            values.append(value)
        
        return np.array(x), np.array(y), np.array(z), np.array(values)
    
    @staticmethod
    def get_2d_vis_data(V, scale, top_left, resolution=1):
        """
        Pull a subset of points from the array V to use in a 2D visualization.
        """
        x = []
        y = []
        values = []
        
        for pt, value in np.ndenumerate(V[::resolution,::resolution]):
            loc = ((np.array(pt) * resolution) / scale) + top_left
            x.append(loc[0])
            y.append(loc[1])
            values.append(value)
        
        return np.array(x), np.array(y), np.array(values)
    
    @staticmethod
    def colormesh_3d(sim, size=(10, 10), color_norm="auto", resolution="auto", graph_ax=None):
        """
        Show the potential sim.V as a shaded color-coded map, where negative velocities are red
        and positive velocitues are blue.
        Parameters:
         - sim: EMSimulationSpace3D instance.
         - size: the figure size of the plot.
         - color_norm: the voltage to use as the maximum for the color display, defaults to largest abs(V[i,j,k]).
         - resolution: how many points to skip when sampling in each direction, defaults to 1.
         - graph_ax: existing graph axes to draw on/
        Produces a plot, and shows it if no axes are provided.
        """
        ax = None
        if graph_ax is None:
            plt.figure(figsize=size)
            ax = plt.axes(projection='3d')
        else:
            ax = graph_ax

        if resolution == "auto":
            resolution = 1
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
        """
        Show two-dimensional cross-sections of the potential sim.V as a shaded color-coded map, 
        where negative velocities are red and positive velocitues are blue.
        Parameters:
         - sim: EMSimulationSpace3D instance.
         - ax_loc: tuples of (axis_id, location) to take cross sections at. See EMSimulationSpace2D.from_3d
         - size: the figure size of the plot.
         - color_norm: the voltage to use as the maximum for the color display, defaults to largest abs(V[i,j,k]).
         - resolution: how many points to skip when sampling in each direction, defaults to 1.
         - graph_ax: existing graph axes to draw on/
        Produces a plot, and shows it if no axes are provided.
        """
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
        """
        Show the potential sim.V as a shaded color-coded map, where negative velocities are red
        and positive velocitues are blue.
        Parameters:
         - sim: EMSimulationSpace2D instance.
         - size: the figure size of the plot.
         - color_norm: the voltage to use as the maximum for the color display, defaults to largest abs(V[i,j]).
         - resolution: how many points to skip when sampling in each direction, defaults to 1.
         - graph_ax: existing graph axes to draw on/
        Produces a plot, and shows it if no axes are provided.
        """
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
        """
        Show contour lines of the potential of a two-dimensional simulation.
        Parameters:
         - sim: EMSimulationSpace2D instance.
         - size: the figure size of the plot.
         - graph_ax: existing graph axes to draw on.
        Produces a plot, and shows it if no axes are provided.
        """
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
        """
        Show the electric field E as three-dimensional arrows in space.
        Parameters:
         - sim3d: EMSimulationSpace3D instance.
         - size: the figure size of the plot.
         - resolution: how many points to skip when sampling in each direction, defaults to 1.
         - graph_ax: existing graph axes to draw on/
        Produces a plot, and shows it if no axes are provided.
        """
        ax = None
        if graph_ax is None:
            plt.figure(figsize=size)
            ax = plt.axes(projection='3d')
        else:
            ax = graph_ax
            
        if resolution == "auto":
            resolution = 1
        
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
        """
        Show the electric field E as two-dimensional arrows in space.
        Parameters:
         - sim2d: EMSimulationSpace2D instance.
         - size: the figure size of the plot.
         - resolution: how many points to skip when sampling in each direction, defaults to 1.
         - graph_ax: existing graph axes to draw on/
        Produces a plot, and shows it if no axes are provided.
        """
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
        """
        Show the trajectory of a particle in three-dimensional space.
        Parameters:
         - time: arrays representing N time indices.
         - x: 3xN array of particle X, Y, Z position
         - size: the figure size of the plot.
         - graph_ax: existing graph axes to draw on/
        Produces a plot, and shows it if no axes are provided.
        """
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
        """
        Show the trajectory of a particle in two-dimensional space.
        Parameters:
         - time: arrays representing N time indices.
         - x: 3xN array of particle X, Y, Z position
         - axis: which component of the motion to not display (slice across).
         - size: the figure size of the plot.
         - graph_ax: existing graph axes to draw on/
        Produces a plot, and shows it if no axes are provided.
        """
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
        """
        Utility to create a video from successive Matplotlib figures.
        Parameters:
         - figure: the matplotlib Figure object which is updated.
         - axes: the collection of matplotlib Axes objects that are drawn to.
         - videoDir: the temporary directory to store video frames in and export to.
         - framerate: how many frames per second.
        """
        self.fig = figure
        self.axes = np.array(axes)
        self.framerate = framerate
        
        self.curr_frame = -1
        
        self.videoDir = "video_tmp" if videoDir is None else videoDir
        if os.path.exists(self.videoDir):
            rmtree(self.videoDir)
        os.mkdir(self.videoDir)
        
    def new_frame(self):
        """
        Called before the plot is updated. Clears the axes.
        """
        self.curr_frame += 1
        for axis in self.axes.flatten():
            axis.clear()
            
    def draw_frame(self, save=True):
        """
        Called after the plot is updated. Draws the canvas and saves the frame to a file.
        Parameter:
         - save: whether to save the file.
        """
        self.fig.canvas.draw()
        if save:
            plt.savefig(os.path.join(self.videoDir, "frame{:03d}.png".format(self.curr_frame)))
        
    def make_movie(self, name="movie.mp4"):
        """
        Invokes FFMPEG to generate a mp4 file from the frames.
        """
        cwd = os.getcwd()
        os.chdir(self.videoDir)
        os.system("ffmpeg -framerate {} -i frame%03d.png -r 24 -pix_fmt yuv420p {}".format(self.framerate, name))
        os.chdir(cwd)
        