import numpy as np
from numba import jit
from scipy.integrate import odeint

GAIN = 1

# Jacobi computation function (3D)
@jit
def jacobi_3d(V, boundary_mask, space, n_points, scale):
    dV = 0
    v_old = 0
    for point, value in np.ndenumerate(V):
        if (point[0] == 0 or point[1] == 0 or point[2] == 0) or \
            (point[0] == space[0]-1 or point[1] == space[1]-1 or point[2] == space[2]-1):
            continue
        else:
            i, j, k = point
            v_old = V[point]
            V[point] = (V[i+1,j,k] + V[i-1,j,k] + 
                        V[i,j+1,k] + V[i,j-1,k] + 
                        V[i,j,k+1] + V[i,j,k-1]) / 6
            if not boundary_mask[point] == 1:
                dV = max(dV, abs(V[point] - v_old))
    return V, dV

# Jacobi computation function (2D)
@jit
def jacobi_2d(V, boundary_mask, space, n_points, scale):
    dV = 0
    v_old = 0
    for point, value in np.ndenumerate(V):
        if (point[0] == 0 or point[1] == 0) or \
            (point[0] == space[0]-1 or point[1] == space[1]-1):
            continue
        else:
            i, j = point
            v_old = V[point]
            V[point] = (V[i+1,j] + V[i-1,j] + 
                        V[i,j+1] + V[i,j-1]) / 4
            if not boundary_mask[point] == 1:
                dV = max(dV, abs(V[point] - v_old))
    return V, dV

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
        
    # Unit conversion functions
    
    def point_to_unit(self, point):
        return tuple([p / self.scale for p in point])
    
    def point_to_global_unit(self, point):
        return tuple([self.top_left[i] + p for i, p in enumerate(self.point_to_unit(point))])
    
    def unit_to_point(self, units):
        return tuple([int(u * self.scale) for u in units])
    
    def global_unit_to_point(self, units):
        return self.unit_to_point([u - self.top_left[i] for i, u in enumerate(units)])
    
    # E Field
    def get_efield(self):
        self.E = -1 * np.array(np.gradient(self.V, 2))
        return self.E
    
    # Jacobi function
    def jacobi(self):
        return inf
    
    # Get the potential value at a specific location
    def E_at(self, location):
        pass
    
    # Computation step
    def compute_step(self, boundary_enforcer=None):
        dV = self.jacobi()
        if boundary_enforcer != None:
            boundary_enforcer(self)
        return dV
    
    # Full computation
    def compute(self, boundary_enforcer=None, convergence_limit=1e-6, transient_ignore=100, maximum_iter=1e6):
        self.boundary_mask = np.zeros(self.V.shape, int)
        if boundary_enforcer != None:
            self.V = np.full(self.V.shape, np.inf)
            boundary_enforcer(self)
            np.isfinite(self.V, self.boundary_mask, where=1)
            self.V = np.zeros(self.V.shape, float)
            
        dV = 10 * convergence_limit
        transient = 0
        while (transient < transient_ignore or dV > convergence_limit) and transient < maximum_iter:
            dV = self.compute_step(boundary_enforcer)
            transient += 1
        print("Computed in", transient, "iterations.")
        return self.V
    

class EMSimulationSpace3D(EMSimulationSpace):
    def __init__(self, space_size=(10, 10, 10), scale=10, top_left=(0, 0, 0), axis_names=("x", "y", "z")):
        if len(space_size) != 3:
            raise ValueError("Space size must be 3D")
        EMSimulationSpace.__init__(self, space_size, scale, top_left, axis_names)
        self.c = 0
        
    def jacobi(self):
        self.V, dV = jacobi_3d(self.V, self.boundary_mask, self.point_space_size, self.n_points, self.scale)
        return dV
    
    def E_at(self, location):
        location = np.array(location)
        i, j, k = self.global_unit_to_point(location)
        if i == 0 or j == 0 or k == 0 or \
            i == self.point_space_size[0]-1 or j == self.point_space_size[1]-1 or k == self.point_space_size[2]-1:
            i = min(max(i, 0), self.point_space_size[0]-1)
            j = min(max(j, 0), self.point_space_size[1]-1)
            k = min(max(k, 0), self.point_space_size[2]-1)
            return self.E[:,i,j,k]
        E = np.zeros(3, float)
        for ax in range(3):
            idsum = 0
            for close_point, v in np.ndenumerate(self.E[ax,i-1:i+2,j-1:j+2,k-1:k+2]):
                close_loc = ((np.array(close_point) + np.array([i-1, j-1, k-1])) / self.scale) + self.top_left
                dist = ((close_loc[0]-location[0])**2 + (close_loc[1]-location[1])**2 + (close_loc[2]-location[2])**2)**0.5
                if dist < 1e-6:
                    return self.E[:,close_point[0],close_point[1],close_point[2]]
                idsum += 1.0 / dist
                E[ax] += (1.0 / dist) * v
            E[ax] /= (idsum * 27)
        return E   
        

class EMSimulationSpace2D(EMSimulationSpace):
    def __init__(self, space_size=(10, 10), scale=10, top_left=(0, 0), axis_names=("x, y")):
        if len(space_size) != 2:
            raise ValueError("Space size must be 2D")
        EMSimulationSpace.__init__(self, space_size, scale, top_left, axis_names)
        
    def jacobi(self):
        self.V, dV = jacobi_2d(self.V, self.boundary_mask, self.point_space_size, self.n_points)
        return dV
    
    def E_at(self, location):
        location = np.array(location)
        i, j = self.global_unit_to_point(location)
        if i == 0 or j == 0 or \
            i == self.point_space_size[0]-1 or j == self.point_space_size[1]-1:
            i = min(max(i, 0), self.point_space_size[0]-1)
            j = min(max(j, 0), self.point_space_size[1]-1)
            return self.E[:,i,j]
        value = 0
        E = np.zeros(2, float)
        for ax in range(2):
            for close_point, v in np.ndenumerate(self.E[ax,i-1:i+2,j-1:j+2]):
                dist = np.linalg.norm(((np.array(close_point) + np.array([i-1, j-1])) / self.scale) - location)
                E[ax] += (1 - dist) * v
            E[ax] /= 9
        return E
    
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
        return sim2d

class ChargedParticle(object):
    GRAVITY = 9.8
    
    def __init__(self, sim, mass, charge, location, velocity, gravity=-1):
        """Initialize a charged particle given a simulation space, mass, charge, position, and velocity."""
        self.sim = sim
        self.mass = mass
        self.charge = charge
        
        if self.sim.dimensions != len(location):
            raise ValueError("Simulation space dimensions must match initial condition.")
        self.initial_location = location
        self.initial_velocity = velocity
        
        self.gravity_axis = gravity
        
    # Equation of motion for the particle
    def eom(self, y, t):
        pass
    
    # Solve equation of motion
    def compute_motion(self, time_range):
        return odeint(self.eom, np.ravel([self.initial_location, self.initial_velocity]), time_range, hmax=(1.0 / self.sim.scale))
    

class ChargedParticle3D(ChargedParticle):
    def __init__(self, sim, mass, charge, location, velocity, gravity=-1):
        ChargedParticle.__init__(self, sim, mass, charge, location, velocity, gravity)           
    
    def eom(self, y, t):
        x = np.array([y[0], y[1], y[2]])
        v = np.array([y[3], y[4], y[5]])
        if self.gravity_axis > 0:
            return np.ravel([v, ((self.charge / self.mass) * self.sim.E_at(x)) - ChargedParticle.GRAVITY])
        return np.ravel([v, (self.charge / self.mass) * self.sim.E_at(x)])
    
    def compute_motion(self, time_range):
        solution = ChargedParticle.compute_motion(self, time_range).T
        self.position = np.array([solution[0], solution[1], solution[2]])
        self.velocity = np.array([solution[3], solution[4], solution[5]])
        self.time = time_range
        return self.position, self.velocity
    
    
        