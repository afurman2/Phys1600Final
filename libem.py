import numpy as np
from numba import jit
from scipy.integrate import solve_ivp

"""
The Electrodynamics Simulation Library libem.py
Author: Adam Furman
Brown University

Provides a simulation framework for computing electric potential,
electric field, and particle motion in three and two dimensions.
"""

@jit
def gs_3d(V, boundary_mask):
    """
    Computation step for the potential on a 3D lattice of points V.
    Parameters:
     - V, a 3D Numpy array of points at which to compute the potential
     - boundary_mask, a Numpy array of the same shape as V with 1s at lattice points containing solid structure.
    Returns the array V with computed voltages and dV, the largest difference between the old and new values.
    """
    dV = 0
    v_old = 0
    for point, value in np.ndenumerate(V):
        if (point[0] == 0 or point[1] == 0 or point[2] == 0) or \
            (point[0] == V.shape[0]-1 or point[1] == V.shape[1]-1 or point[2] == V.shape[2]-1):
            # The point is on the boundary. Keep initial value.
            continue
        else:
            i, j, k = point
            v_old = V[point]
            V[point] = (V[i+1,j,k] + V[i-1,j,k] + 
                        V[i,j+1,k] + V[i,j-1,k] + 
                        V[i,j,k+1] + V[i,j,k-1]) / 6
            if not boundary_mask[point] == 1:
                # Ignore boundary conditions when looking for error
                dV = max(dV, abs(V[point] - v_old))
    return V, dV

@jit
def gs_2d(V, boundary_mask):
    """
    Computation step for the potential on a 2D lattice of points V.
    Parameters:
     - V, a 2D Numpy array of points at which to compute the potential
     - boundary_mask, a Numpy array of the same shape as V with 1s at lattice points containing solid structure.
    Returns the array V with computed voltages and dV, the largest difference between the old and new values.
    """
    dV = 0
    v_old = 0
    for point, value in np.ndenumerate(V):
        if (point[0] == 0 or point[1] == 0) or \
            (point[0] == V.shape[0]-1 or point[1] == V.shape[1]-1):
            # The point is on the boundary. Keep initial value.
            continue
        else:
            i, j = point
            v_old = V[point]
            V[point] = (V[i+1,j] + V[i-1,j] + 
                        V[i,j+1] + V[i,j-1]) / 4
            if not boundary_mask[point] == 1:
                # Ignore boundary conditions when looking for error
                dV = max(dV, abs(V[point] - v_old))
    return V, dV

class EMSimulationSpace(object):
    def __init__(self, space_size, scale, top_left, axis_names=None):
        """
        Initialize a generic EM simulation space. The space primarily consists of the V array,
        which is a square lattice of nodes where the voltage is computed.
        Parameters:
         - space_size: tuple of the dimensions of the space in simulation units.
         - scale: the number of simulation lattice points per linear simulation unit.
         - top_left: the location in simulation units corresponding to the zeroth lattice point.
         - axis_names: the names of the axes in order.
        """
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
        
    @staticmethod
    def load(filepath):
        """
        Loads a simulation space from the file stored at <filepath>.
        """
        data = np.load(filepath, allow_pickle=True)
        space = None
        if len(data[0]) == 3:
            space = EMSimulationSpace3D(data[0], data[1], data[2], data[3])
        elif len(data[0]) == 2:
            space = EMSimulationSpace2D(data[0], data[1], data[2], data[3])
        else:
            space = EMSimulationSpace(data[0], data[1], data[2], data[3])
        space.V = data[4]
        space.boundary_mask = data[5]
        space.get_efield()
        return space
        
    def save(self, filepath):
        """
        Save the simulation space to <filepath>.
        """
        np.save(filepath, np.array([self.space_size, self.scale, self.top_left, self.axis_names,
                                    self.V, self.boundary_mask], dtype=object))
            
    def point_to_unit(self, point):
        """
        Convert a simulation lattice point to a relative simulation unit.
        """
        return tuple([p / self.scale for p in point])
    
    def point_to_global_unit(self, point):
        """
        Convert a simulation lattice point to a global simulation unit.
        """
        return tuple([self.top_left[i] + (p / self.scale) for i, p in enumerate(point)])
    
    def unit_to_point(self, units):
        """
        Convert a relative unit to a simulation lattice point.
        """
        return tuple([round(u * self.scale) for u in units])
    
    def global_unit_to_point(self, units):
        """
        Convert an absolute unit to a simulation lattice point.
        """
        return self.unit_to_point([u - self.top_left[i] for i, u in enumerate(units)])
    
    def get_efield(self):
        """
        Computes the electric field E from the potential V.
        Returns a Numpy array of shape dimensions x V.shape.
        """
        self.E = -1 * np.array(np.gradient(self.V, 1.0 / self.scale, edge_order=2))
        return self.E
    
    def gauss_seidel(self):
        """
        Compute the potential using the Gauss-Seidel method.
        Returns the error dV of the step.
        """
        return inf
    
    def E_at(self, location):
        """
        Returns the electric field at a given location in simulation units.
        """
        pass
    
    def compute_step(self, boundary_enforcer=None):
        """
        Run one step of the Gauss-Seidel computation and enforce the boundary conditions.
        Parameters:
         - boundary_enforcer [None or callable]: a function which imposes boundary conditions given the simulation space.
        Returns the error dV of the step.
        """
        dV = self.gauss_seidel()
        if boundary_enforcer != None:
            boundary_enforcer(self)
        return dV
    
    def compute(self, boundary_enforcer=None, convergence_limit=1e-6, transient_ignore=100, maximum_iter=1e6, debug=False):
        """
        Compute the potential from boundary conditions.
        Parameters:
         - boundary_enforcer [None or callable]: a function which imposes boundary conditions given the simulation space.
         - convergence_limit: The step error dV that needs to be reached for the computation to stop.
         - transient_ignore: The minimum number of steps to compute.
         - maximum_iter: The maximum number of steps to compute.
         - debug: Whether to print a completion message with the number of steps taken.
        Returns the computed potential V.
        """
        self.boundary_mask = np.zeros(self.V.shape, int)
        if boundary_enforcer != None:
            # Determine where the solid objects are.
            self.V = np.full(self.V.shape, np.inf)
            boundary_enforcer(self)
            np.isfinite(self.V, self.boundary_mask, where=1)
            self.V = np.zeros(self.V.shape, float)
            
        dV = 10 * convergence_limit
        transient = 0
        while (transient < transient_ignore or dV > convergence_limit) and transient < maximum_iter:
            dV = self.compute_step(boundary_enforcer)
            transient += 1
        if debug:
            print("Computed in", transient, "iterations.")
        return self.V
    
    def detect_hit(self, position, velocity, radius=0):
        """
        Used to determine whether a particle will hit a solid part of the simulated space.
        Parameters:
         - position: the location of the particle in absolute simulation units.
         - velocity: the velocity of the particle in simulation units.
         - radius: the size of the particle in simulation units.
        """
        return_v = np.zeros(len(velocity), float)
        # Check the velocity in each unit direction
        for axis, component in enumerate(velocity):
            step = np.zeros(len(velocity), float)
            # Step forward in the specified direction
            step[axis] = radius + (velocity[axis] / self.scale)
            try:
                if self.boundary_mask[self.global_unit_to_point(position + step)] == 1:
                    # The particle bounces in this direction. Flip the respective component.
                    return_v[axis] = -velocity[axis]
            except (IndexError, ValueError):
                # The particle is out of bounds. Return zero.
                return np.zeros(len(velocity), float)
        return return_v
    

class EMSimulationSpace3D(EMSimulationSpace):
    def __init__(self, space_size=(10, 10, 10), scale=10, top_left=(0, 0, 0), axis_names=("x", "y", "z")):
        """
        Three-dimensional implementation of the Simulation Space.
        See documentation EMSimulationSpace.
        Requires the size of the space to have dimension 3.
        """
        if len(space_size) != 3:
            raise ValueError("Space size must be 3D")
        EMSimulationSpace.__init__(self, space_size, scale, top_left, axis_names)
        self.c = 0
        
    def gauss_seidel(self):
        """
        Compute the potential using the three-dimensional Gauss-Seidel method.
        Returns the error dV of the step.
        """
        self.V, dV = gs_3d(self.V, self.boundary_mask)
        return dV
    
    def E_at(self, location):
        """
        Parameters:
         - location in simulation units.
        Returns the electric field at a given location in simulation units.
        """
        location = np.array(location)
        i, j, k = self.global_unit_to_point(location)
        if i <= 1 or j <= 1 or k <= 1 or \
            i >= self.point_space_size[0]-2 or j >= self.point_space_size[1]-2 or k >= self.point_space_size[2]-2:
            # The particle is out of bounds. Return closest available point.
            i = min(max(i, 0), self.point_space_size[0]-1)
            j = min(max(j, 0), self.point_space_size[1]-1)
            k = min(max(k, 0), self.point_space_size[2]-1)
            return self.E[:,i,j,k]
        
        if np.linalg.norm(location - np.array(self.point_to_global_unit((i, j, k)))) < 1e-4:
            # The particle is immediately on top of a simulated point. Return value there.
            return self.E[:,i,j,k]
                
        E = np.zeros(3, float)
        
        # Compute the inverse distance weighted average of E field at nearby points.
        values = np.copy(self.E[:,i-1:i+2,j-1:j+2,k-1:k+2])
        close_points = np.indices(values.shape[1:]).astype(float)
        close_points[:,:,:] += np.array([i-1, j-1, k-1])
        close_locations = (close_points / self.scale) + self.top_left
        
        inv_distances = 1.0 / np.linalg.norm(location - close_locations[:,:,:], axis=3)
        
        values *= inv_distances
                        
        return np.sum(values, axis=(1, 2, 3)) / np.sum(inv_distances)
        

class EMSimulationSpace2D(EMSimulationSpace):
    def __init__(self, space_size=(10, 10), scale=10, top_left=(0, 0), axis_names=("x, y")):
        """
        Two-dimensional implementation of the Simulation Space.
        See documentation EMSimulationSpace.
        Requires the size of the space to have dimension 2.
        """
        if len(space_size) != 2:
            raise ValueError("Space size must be 2D")
        EMSimulationSpace.__init__(self, space_size, scale, top_left, axis_names)
        
    def gauss_seidel(self):
        """
        Compute the potential using the two-dimensional Gauss-Seidel method.
        Returns the error dV of the step.
        """
        self.V, dV = gs_2d(self.V, self.boundary_mask)
        return dV
        
    def E_at(self, location):
        """
        Parameters:
         - location in simulation units.
        Returns the electric field at a given location in simulation units.
        """
        location = np.array(location)
        i, j = self.global_unit_to_point(location)
        if i <= 1 or j <= 1 or \
            i >= self.point_space_size[0]-2 or j >= self.point_space_size[1]-2:
            # The particle is out of bounds. Return closest available point.
            i = min(max(i, 0), self.point_space_size[0]-1)
            j = min(max(j, 0), self.point_space_size[1]-1)
            return self.E[:,i,j]
        
        if np.linalg.norm(location - np.array(self.point_to_global_unit((i, j)))) < 1e-4:
            # The particle is immediately on top of a simulated point. Return value there.
            return self.E[:,i,j]
                
        E = np.zeros(2, float)
        
        # Compute the inverse distance weighted average of E field at nearby points.
        values = np.copy(self.E[:,i-1:i+2,j-1:j+2])
        close_points = np.indices(values.shape[1:]).astype(float)
        close_points[:,:] += np.array([i-1, j-1])
        close_locations = (close_points / self.scale) + self.top_left
        
        inv_distances = 1.0 / np.linalg.norm(location - close_locations[:,:], axis=2)
        
        values *= inv_distances
                        
        return np.sum(values, axis=(1, 2)) / np.sum(inv_distances)
    
    def get_meshgrid(self):
        """
        Generate a MeshGrid in simulation units mapping to the simulation lattice points.
        """
        return np.meshgrid([self.top_left[0] + (i / self.scale) for i in range(self.point_space_size[0])],
                          [self.top_left[1] + (i / self.scale) for i in range(self.point_space_size[1])])
        
    @staticmethod
    def from_3d(sim3d, axis=0, location=0):
        """
        Generate a two-dimensional simulation space as a "slice" of a three-dimensional one.
        Parameters:
         - sim3d: an EMSimulationSpace3D instance.
         - axis: (x is 0, y is 1, z is 2) to remove in the slice.
         - location: the location along the specified <axis> to slice at.
        Returns an EMSimulationSpace2D instance.
        """
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
    
    def __init__(self, sim, mass, charge, location, velocity, radius=0, gravity=-1, bounce=None, track_force=False):
        """
        Initialize a generic ChargedParticle instance representing a particle with mass and charge.
        Parameters:
         - sim: the simulation space the particle moves in.
         - mass: the mass of the particle in simulation units.
         - charge: the charge of the particle in simulation units.
         - location: the initial location of the particle in simulation units.
         - velocity: the initial velocity of the particle in simulation units.
         - radius: the size of the particle in simulation units.
         - gravity: the axis along which to apply GRAVITY, or -1 for off.
         - bounce: the fraction of velocity to keep when bouncing, or None for no bounces.
         - track_force: whether to store force when running computations.
        """
        self.sim = sim
        self.mass = mass
        self.charge = charge
        self.radius = radius
        
        self.initial_location = np.array(location)
        self.initial_velocity = np.array(velocity)
        
        self.gravity = np.zeros(len(location), float)
        if gravity != -1:
            self.gravity[gravity] = ChargedParticle.GRAVITY
        self.bounce = bounce
        self.track_force = track_force
        
    @staticmethod
    def make_terminating_function(method):
        """
        Helper method that adds the .terminal = True property to a method.
        See scipy solve_ivp.
        Parameters:
         - method [callable]
        Returns function with .terminal = True property set.
        """
        def runner(t, y):
            return method(t, y)
        runner.terminal = True
        return runner
        
    def eom(self, t, y):
        """
        Equation of motion for the particle.
        Parameters:
         - t: time in simulation units.
         - y: [x, y, z, vx, vy, vz] in simulation units.
        Returns increment [vx, vy, vz, Fx, Fy, Fz] in simulation units.
        """
        pass
    
    def collision_event(self, t, y):
        """
        Parameters:
         - t: time in simulation units.
         - y: [x, y, z, vx, vy, vz] in simulation units.
        Returns -1 if no collision is detected, +1 if it is.
        """
        return -1.0
    
    def compute_motion(self, t_span, stop_cond=None):
        """
        Computes the motion of the particle in the given time range. Can stop sooner.
        Automatically adds bounce detector if the particle's bounce coefficient is not None.
        Parameters:
         - t_span: tuple (t_0, t1) of time in simulation units.
         - stop_cond [callable or None]: function to use as a terminating condition (see solve_ivp).
        Returns output of solve_ivp.
        """
        term_event = ChargedParticle.make_terminating_function(self.collision_event)
        if not stop_cond is None:
            stop_cond = ChargedParticle.make_terminating_function(stop_cond)
        stop_events = ([term_event] if not self.bounce is None else []) + ([stop_cond] if not stop_cond is None else [])
        return solve_ivp(self.eom, t_span,
                         y0=np.ravel([self.initial_location, self.initial_velocity]),
                         method="DOP853", max_step=(min(self.sim.space_size) / 20.0),
                         events=stop_events)
    

class ChargedParticle3D(ChargedParticle):
    def __init__(self, sim, mass, charge, location, velocity, radius=0, gravity=-1, bounce=None, track_force=False):
        """
        Three-dimensional implementation of the ChrargedParticle.
        See documentation there.
        """
        ChargedParticle.__init__(self, sim, mass, charge, location, velocity, radius, gravity, bounce, track_force)
        self.bounce_velocity = None
        self.num_bounces = 0
        
    def collision_event(self, t, y):
        """
        Parameters:
         - t: time in simulation units.
         - y: [x, y, z, vx, vy, vz] in simulation units.
        Returns -1 if no collision is detected, +1 if it is.
        Sets the bounce velocity to the output of the bounce detector.
        """
        x = np.array([y[0], y[1], y[2]])
        v = np.array([y[3], y[4], y[5]])
        hit_v = self.sim.detect_hit(x, v, self.radius)
        if hit_v.any():
            # One time step in any direction leads to a bounce.
            self.bounce_velocity = v + 2 * (hit_v * self.bounce)
            return 1.0
        return -1.0
    
    def eom(self, t, y):
        """
        Equation of motion for the particle.
        Parameters:
         - t: time in simulation units.
         - y: [x, y, z, vx, vy, vz] in simulation units.
        Returns increment [vx, vy, vz, Fx, Fy, Fz] in simulation units.
        Applies gravity if it is specified.
        """
        x = np.array([y[0], y[1], y[2]])
        v = np.array([y[3], y[4], y[5]])
        F = ((self.charge / self.mass) * self.sim.E_at(x)) - self.gravity
        if self.track_force and not t in self.force:
            self.force[t] = F
        return np.ravel([v, F])
    
    def compute_motion(self, t_span, stop_cond=None):
        """
        Computes the motion of the particle in the given time range. Can stop sooner.
        Automatically adds bounce detector if the particle's bounce coefficient is not None.
        Parameters:
         - t_span: tuple (t_0, t1) of time in simulation units.
         - stop_cond [callable or None]: function to use as a terminating condition (see solve_ivp).
        Returns output the particle's location and velocity history.
        """
        initial_v = np.copy(self.initial_velocity)
        initial_l = np.copy(self.initial_location)
        
        if self.track_force:
            self.force = {}
        
        total_time = t_span[0]
        time_partial = []
        position_partial = []
        velocity_partial = []
        
        while total_time < t_span[1]:
            try:
                # Compute over the remaining time interval.
                p_res = ChargedParticle.compute_motion(self, (total_time, t_span[1]), stop_cond)
                time_partial.append(p_res.t)
                position_partial.append(np.array([p_res.y[0], p_res.y[1], p_res.y[2]]))
                velocity_partial.append(np.array([p_res.y[3], p_res.y[4], p_res.y[5]]))
                total_time = p_res.t[-1]
                if not self.bounce_velocity is None:
                    # The particle bounced, set the initial conditions to the location of the bounce.
                    self.initial_velocity = self.bounce_velocity
                    self.initial_location = position_partial[-1][:,-1]
                    self.num_bounces += 1
            except Exception as e:
                print("Exception occured during time step", (total_time, t_span[1]), ":", e)
                break
            
        self.time = np.concatenate(time_partial)
        self.position = np.concatenate(position_partial, axis=1)
        self.velocity = np.concatenate(velocity_partial, axis=1)
        if self.track_force:
            self.force = np.stack([self.force[t] for t in self.time], axis=0)
        # Reset the state to proper initial conditions
        self.initial_velocity = initial_v
        self.initial_location = initial_l
        
        return self.position, self.velocity
    
    @staticmethod
    def generate_particles(n, sim, m_avg, q_avg, l0_avg, v0_avg,
                              m_std=0, q_std=0, l0_std=(0, 0, 0), v0_std=(0, 0, 0),
                               bounce_coef=None, track_f=False):
        """
        Generate a set of particles with normally-distributed properties.
        Parameters:
         - n: number of particles to generate.
         - sim: the simulation space to initialize them in.
         - m_avg... m_std: the mass average and st. dev.
         - q_avg... q_std: the charge average and st. dev.
         - l0_avg... l0_std: the initial location average and st. dev. as tuples.
         - v0_avg... v0_std: the ititial velocity average and st. dev. as tuples.
         - bounce_coef: the bounce coefficient for all particles.
         - track_f: whether to track force for all particles.
        Returns a list of n ChargedParticle3D instances.
        """
        particles = []
        for _ in range(n):
            mass = np.random.normal(loc=m_avg, scale=m_std)
            charge = np.random.normal(loc=q_avg, scale=q_std)
            loc = [np.random.normal(loc=l0_avg[i], scale=l0_std[i]) for i in range(len(l0_avg))]
            vel = [np.random.normal(loc=v0_avg[i], scale=v0_std[i]) for i in range(len(v0_avg))]
            
            particles.append(ChargedParticle3D(sim, mass, charge, loc, vel, bounce=bounce_coef, track_force=track_f))
            
        return particles
        