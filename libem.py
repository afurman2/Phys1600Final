import numpy as np
from numba import jit

# Jacobi computation function (3D)
@jit
def jacobi_3d(V, V_new, space, n_points):
    for point, value in np.ndenumerate(V):
        if (point[0] == 0 or point[1] == 0 or point[2] == 0) or \
            (point[0] == space[0]-1 or point[1]-1 == space[1] or point[2] == space[2]-1):
            V_new[point] = V[point]
        else:
            i, j, k = point
            V_new[point] = (V[i+1,j,k] + V[i-1,j,k] + 
                            V[i,j+1,k] + V[i,j-1,k] + 
                            V[i,j,k+1] + V[i,j,k-1]) / 6
    dV = np.sum(np.abs(V_new - V)) / n_points
    return V_new, dV

# Jacobi computation function (2D)
@jit
def jacobi_2d(V, V_new, space, n_points):
    for point, value in np.ndenumerate(V):
        if (0 in point) or \
            (point[0] == space[0]-1 or point[1] == space[1]-1):
            V_new[point] = V[point]
        else:
            i, j = point
            V_new[point] = (V[i+1,j] + V[i-1,j] + 
                            V[i,j+1] + V[i,j-1]) / 4
    dV = np.sum(np.abs(V_new - V)) / n_points
    V = np.copy(V_new)
    return V_new, dV

class EMSimulationSpace(object):
    def __init__(self, space_size, scale, top_left, axis_names=None):
        """Initialize a simulation space given an array of sizes in units and conversion factor (points / unit)"""
        if len(space_size) != len(top_left):
            raise ValueError("The dimensions of top_left need to match space_size")
            
        self.space_size = np.array(space_size)
        self.scale = scale
        self.top_left = np.array(top_left)
        self.axis_names = [str(i + 1) for i in range(len(space_size))] if axis_names is None else axis_names
        
        self.dimensions = len(space_size)
        self.point_space_size = self.space_size * self.scale
        self.n_points = np.prod(self.point_space_size)
        
        self.V = np.zeros([s * self.scale for s in self.space_size])
        self.V_new = np.copy(self.V)
        
    # Unit conversion functions
    
    def point_to_unit(self, point):
        return tuple([p / self.scale for p in point])
    
    def point_to_global_unit(self, point):
        return tuple([self.top_left[i] + p for i, p in enumerate(self.point_to_unit(point))])
    
    def unit_to_point(self, units):
        return tuple([int(u * self.scale) for u in units])
    
    def global_unit_to_point(self, units):
        return self.unit_to_point([u - self.top_left[i] for i, u in enumerate(units)])
    
    # Gradient
    def get_gradient(self):
        return np.gradient(self.V, 2)
    
    # Jacobi function
    def jacobi(self):
        return inf
    
    # Computation step
    def compute_step(self, boundary_enforcer=None):
        dV = self.jacobi()
        if boundary_enforcer != None:
            boundary_enforcer(self)
        return dV
    
    # Full computation
    def compute(self, boundary_enforcer=None, convergence_limit=1e-6, transient_ignore=100):
        dV = 10 * convergence_limit
        transient = 0
        while transient < transient_ignore or dV > convergence_limit:
            dV = self.compute_step(boundary_enforcer)
            transient += 1
        print("Computed in", transient, "iterations.")
        return self.V

class EMSimulationSpace3D(EMSimulationSpace):
    def __init__(self, space_size=(10, 10, 10), scale=10, top_left=(0, 0, 0), axis_names=("x", "y", "z")):
        if len(space_size) != 3:
            raise ValueError("Space size must be 3D")
        EMSimulationSpace.__init__(self, space_size, scale, top_left, axis_names)
        
    def jacobi(self):
        self.V, dV = jacobi_3d(self.V, self.V_new, self.point_space_size, self.n_points)
        return dV

class EMSimulationSpace2D(EMSimulationSpace):
    def __init__(self, space_size=(10, 10), scale=10, top_left=(0, 0), axis_names=("x, y")):
        if len(space_size) != 2:
            raise ValueError("Space size must be 2D")
        EMSimulationSpace.__init__(self, space_size, scale, top_left, axis_names)
        
    def jacobi(self):
        self.V, dV = jacobi_2d(self.V, self.V_new, self.point_space_size, self.n_points)
        return dV
    
    # Get meshgrid for plotting
    def get_meshgrid(self):
        return np.meshgrid([self.top_left[0] + (i / self.scale) for i in range(self.point_space_size[0])],
                          [self.top_left[1] + (i / self.scale) for i in range(self.point_space_size[1])])
        
    # Generate 2d from 3d simulation
    @staticmethod
    def from_3d(sim3d, axis=0, location=0):
        space_size = np.delete(sim3d.space_size, axis)
        top_left = np.delete(sim3d.top_left, axis)
        axis_labels = np.delete(sim3d.axis_names, axis)
        sim2d = EMSimulationSpace2D(space_size, sim3d.scale, top_left, axis_labels)
        if axis == 0:
            loc = sim3d.global_unit_to_point((location, 0, 0))
            sim2d.V = sim3d.V[loc[0],:,:]
        elif axis == 1:
            loc = sim3d.global_unit_to_point((0, location, 0))
            sim2d.V = sim3d.V[:,loc[1],:]
        elif axis == 2:
            loc = sim3d.global_unit_to_point((0, 0, location))
            sim2d.V = sim3d.V[:,:,loc[2]]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")
        sim2d.V_new = np.copy(sim2d.V)
        return sim2d
