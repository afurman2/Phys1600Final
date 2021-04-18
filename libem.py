import numpy as np
from numba import jit
from scipy.integrate import solve_ivp

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
        
    # Load from save
    @staticmethod
    def load(filepath):
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
        
    # Save
    def save(self, filepath):
        np.save(filepath, np.array([self.space_size, self.scale, self.top_left, self.axis_names,
                                    self.V, self.boundary_mask], dtype=object))
        
    # Unit conversion functions
    
    def point_to_unit(self, point):
        return tuple([p / self.scale for p in point])
    
    def point_to_global_unit(self, point):
        return tuple([self.top_left[i] + (p / self.scale) for i, p in enumerate(point)])
    
    def unit_to_point(self, units):
        return tuple([round(u * self.scale) for u in units])
    
    def global_unit_to_point(self, units):
        return self.unit_to_point([u - self.top_left[i] for i, u in enumerate(units)])
    
    # E Field
    def get_efield(self):
        self.E = -1 * np.array(np.gradient(self.V, 1.0 / self.scale, edge_order=2))
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
    def compute(self, boundary_enforcer=None, convergence_limit=1e-6, transient_ignore=100, maximum_iter=1e6, debug=False):
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
        if debug:
            print("Computed in", transient, "iterations.")
        return self.V
    
    # Detect hit of solid object
    def detect_hit(self, position, velocity, radius=0):
        return_v = np.zeros(len(velocity), float)
        for axis, component in enumerate(velocity):
            step = np.zeros(len(velocity), float)
            step[axis] = radius + (velocity[axis] / self.scale)
            try:
                if self.boundary_mask[self.global_unit_to_point(position + step)] == 1:
                    return_v[axis] = -velocity[axis]
            except (IndexError, ValueError):
                return np.zeros(len(velocity), float)
        return return_v
    

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
        if i <= 1 or j <= 1 or k <= 1 or \
            i >= self.point_space_size[0]-2 or j >= self.point_space_size[1]-2 or k >= self.point_space_size[2]-2:
            i = min(max(i, 0), self.point_space_size[0]-1)
            j = min(max(j, 0), self.point_space_size[1]-1)
            k = min(max(k, 0), self.point_space_size[2]-1)
            return self.E[:,i,j,k]
        
        if np.linalg.norm(location - np.array(self.point_to_global_unit((i, j, k)))) < 1e-4:
            return self.E[:,i,j,k]
                
        E = np.zeros(3, float)
        
        values = np.copy(self.E[:,i-1:i+2,j-1:j+2,k-1:k+2])
        close_points = np.indices(values.shape[1:]).astype(float)
        close_points[:,:,:] += np.array([i-1, j-1, k-1])
        close_locations = (close_points / self.scale) + self.top_left
        
        inv_distances = 1.0 / np.linalg.norm(location - close_locations[:,:,:], axis=3)
        
        values *= inv_distances
                        
        return np.sum(values, axis=(1, 2, 3)) / np.sum(inv_distances)
        

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
        if i <= 1 or j <= 1 or \
            i >= self.point_space_size[0]-2 or j >= self.point_space_size[1]-2:
            i = min(max(i, 0), self.point_space_size[0]-1)
            j = min(max(j, 0), self.point_space_size[1]-1)
            return self.E[:,i,j]
        
        if np.linalg.norm(location - np.array(self.point_to_global_unit((i, j)))) < 1e-4:
            return self.E[:,i,j]
                
        E = np.zeros(2, float)
        
        values = np.copy(self.E[:,i-1:i+2,j-1:j+2])
        close_points = np.indices(values.shape[1:]).astype(float)
        close_points[:,:] += np.array([i-1, j-1])
        close_locations = (close_points / self.scale) + self.top_left
        
        inv_distances = 1.0 / np.linalg.norm(location - close_locations[:,:], axis=2)
        
        values *= inv_distances
                        
        return np.sum(values, axis=(1, 2)) / np.sum(inv_distances)
    
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
    
    def __init__(self, sim, mass, charge, location, velocity, radius=0, gravity=-1, bounce=None, track_force=False):
        """Initialize a charged particle given a simulation space, mass, charge, position, and velocity."""
        self.sim = sim
        self.mass = mass
        self.charge = charge
        self.radius = radius
        
        if self.sim.dimensions != len(location):
            raise ValueError("Simulation space dimensions must match initial condition.")
        self.initial_location = np.array(location)
        self.initial_velocity = np.array(velocity)
        
        self.gravity = np.zeros(len(location), float)
        if gravity != -1:
            self.gravity[gravity] = ChargedParticle.GRAVITY
        self.bounce = bounce
        self.track_force = track_force
        
    @staticmethod
    def make_terminating_function(method):
        def runner(t, y):
            return method(t, y)
        runner.terminal = True
        return runner
        
    # Equation of motion for the particle
    def eom(self, t, y):
        pass
    
    # Event to detect collisons
    def collision_event(self, t, y):
        return -1.0
    
    # Solve equation of motion
    def compute_motion(self, t_span, stop_cond=None):
        term_event = ChargedParticle.make_terminating_function(self.collision_event)
        if not stop_cond is None:
            stop_cond = ChargedParticle.make_terminating_function(stop_cond)
        stop_events = ([term_event] if not self.bounce is None else []) + ([stop_cond] if not stop_cond is None else [])
        return solve_ivp(self.eom, t_span,
                         y0=np.ravel([self.initial_location, self.initial_velocity]),
                         method="DOP853", max_step=(min(self.sim.space_size) / 10.0),
                         events=stop_events)
    

class ChargedParticle3D(ChargedParticle):
    def __init__(self, sim, mass, charge, location, velocity, radius=0, gravity=-1, bounce=None, track_force=False):
        ChargedParticle.__init__(self, sim, mass, charge, location, velocity, radius, gravity, bounce, track_force)
        self.bounce_velocity = None
        self.num_bounces = 0
        
    def collision_event(self, t, y):
        x = np.array([y[0], y[1], y[2]])
        v = np.array([y[3], y[4], y[5]])
        hit_v = self.sim.detect_hit(x, v, self.radius)
        if hit_v.any():
            self.bounce_velocity = v + 2 * (hit_v * self.bounce)
            return 1.0
        return -1.0
    
    def eom(self, t, y):
        x = np.array([y[0], y[1], y[2]])
        v = np.array([y[3], y[4], y[5]])
        F = ((self.charge / self.mass) * self.sim.E_at(x)) - self.gravity
        if self.track_force and not t in self.force:
            self.force[t] = F
        return np.ravel([v, F])
    
    def compute_motion(self, t_span):        
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
                p_res = ChargedParticle.compute_motion(self, (total_time, t_span[1]))
                time_partial.append(p_res.t)
                position_partial.append(np.array([p_res.y[0], p_res.y[1], p_res.y[2]]))
                velocity_partial.append(np.array([p_res.y[3], p_res.y[4], p_res.y[5]]))
                total_time = p_res.t[-1]
                if not self.bounce_velocity is None:
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
        self.initial_velocity = initial_v
        self.initial_location = initial_l
        
        return self.position, self.velocity
    
    @staticmethod
    def generate_particles(n, sim, m_avg, q_avg, l0_avg, v0_avg,
                              m_std=0, q_std=0, l0_std=(0, 0, 0), v0_std=(0, 0, 0),
                               bounce_coef=None, track_f=False):
        particles = []
        for _ in range(n):
            mass = np.random.normal(loc=m_avg, scale=m_std)
            charge = np.random.normal(loc=q_avg, scale=q_std)
            loc = [np.random.normal(loc=l0_avg[i], scale=l0_std[i]) for i in range(len(l0_avg))]
            vel = [np.random.normal(loc=v0_avg[i], scale=v0_std[i]) for i in range(len(v0_avg))]
            
            particles.append(ChargedParticle3D(sim, mass, charge, loc, vel, bounce=bounce_coef, track_force=track_f))
            
        return particles
        