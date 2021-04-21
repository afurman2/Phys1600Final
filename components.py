import numpy as np

"""
The Electrodynamics Simulation Library components.py
Author: Adam Furman
Brown University

Provides solid object simulations for use with the framework.
"""

def make_enforcer(*lambdas):
    """
    Package a series of boundary conditions into a single callable.
    Parameters:
     * list of [callable](sim)
    Returns [callable](sim)
    """
    return lambda sim: [f(sim) for f in lambdas]

def enf(func, *args, **kwargs):
    """
    Returns a callable representation of enforced boundary conditions.
    Parameters:
     - func [callable]: A boundary enforcing function that takes (sim, *args, **kwargs)
    Returns [callable](sim)
    """
    return lambda sim: func(sim, *args, **kwargs)

class EMObjects:
    """
    All methods meant to be used with the enf(...) tool.
    Specifying a point charge, for example, would be done as:
    make_enforcer(enf(EMObjects.point_charge, (x, y), V_0))
    """
    @staticmethod
    def point_charge(sim, location, voltage):
        """
        A point charge at a <location> in simulation units held at <voltage>.
        """
        sim.V[sim.global_unit_to_point(location)] = voltage
        
    @staticmethod
    def outer_edge_3d(sim, voltage):
        """
        Specify that the outer edges (not faces) of the space should be held at <voltage>.
        """
        sim.V[0,0,:] = voltage
        sim.V[0,sim.point_space_size[1]-1,:] = voltage
        sim.V[sim.point_space_size[0]-1,0,:] = voltage
        sim.V[sim.point_space_size[0]-1,sim.point_space_size[1]-1,:] = voltage
        
        sim.V[0,:,0] = voltage
        sim.V[sim.point_space_size[0]-1,:,0] = voltage
        sim.V[0,:,sim.point_space_size[2]-1] = voltage
        sim.V[sim.point_space_size[0]-1,:,sim.point_space_size[2]-1] = voltage
        
        sim.V[:,0,0] = voltage
        sim.V[:,0,sim.point_space_size[2]-1] = voltage
        sim.V[:,sim.point_space_size[1]-1,0] = voltage
        sim.V[:,sim.point_space_size[1]-1,sim.point_space_size[2]-1] = voltage
        
    @staticmethod
    def outer_plane_3d(sim, voltage):
        """
        Specify that the outer faces of the space should be held at <voltage>.
        """
        sim.V[0,:,:] = voltage
        sim.V[sim.point_space_size[0]-1,:,:] = voltage
        
        sim.V[:,0,:] = voltage
        sim.V[:,sim.point_space_size[1]-1,:] = voltage
        
        sim.V[:,:,0] = voltage
        sim.V[:,:,sim.point_space_size[2]-1] = voltage
        
    @staticmethod
    def rectangular_prism_solid(sim, top_left, lwh, voltage):
        """
        A solid rectagular prism with (l, w, h) measured from <top_left> held at <voltage>.
        Both top_left and lwh should be 3-tuples.
        """
        top_left = sim.global_unit_to_point(top_left)
        lwh = sim.unit_to_point(lwh)
        sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]:top_left[1]+lwh[1],top_left[2]:top_left[2]+lwh[2]] = voltage
        
    @staticmethod
    def rectangular_prism_hollow(sim, top_left, lwh, thickness, voltage):
        """
        A hollow capped rectagular prism with (l, w, h) measured from <top_left> held at <voltage>.
        The prism is <thickness> thick from the outside in (e.g. bounded by lwh).
        Both top_left and lwh should be 3-tuples.
        """
        top_left = sim.global_unit_to_point(top_left)
        lwh = sim.unit_to_point(lwh)
        th = sim.unit_to_point((thickness,))[0]
        
        inside = sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]+th:top_left[2]+lwh[2]-th].copy()
        sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]:top_left[1]+lwh[1],top_left[2]:top_left[2]+lwh[2]] = voltage
        sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]+th:top_left[2]+lwh[2]-th] = inside
        
    @staticmethod
    def rectangular_prism_hollow_nocap(sim, top_left, lwh, thickness, cap_axis, voltage):
        """
        A hollow  rectagular prism with (l, w, h) measured from <top_left> held at <voltage>.
        The prism is <thickness> thick from the outside in (e.g. bounded by lwh).
        The prism is open along <cap_axis>.
        Both top_left and lwh should be 3-tuples.
        """
        top_left = sim.global_unit_to_point(top_left)
        lwh = sim.unit_to_point(lwh)
        th = sim.unit_to_point((thickness,))[0]
        
        if cap_axis == 0:
            inside = sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]+th:top_left[2]+lwh[2]-th].copy()
            sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]:top_left[1]+lwh[1],top_left[2]:top_left[2]+lwh[2]] = voltage
            sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]+th:top_left[2]+lwh[2]-th] = inside
        elif cap_axis == 1:
            inside = sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]:top_left[1]+lwh[1],top_left[2]+th:top_left[2]+lwh[2]-th].copy()
            sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]:top_left[1]+lwh[1],top_left[2]:top_left[2]+lwh[2]] = voltage
            sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]:top_left[1]+lwh[1],top_left[2]+th:top_left[2]+lwh[2]-th] = inside
        elif cap_axis == 2:
            inside = sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]:top_left[2]+lwh[2]].copy()
            sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]:top_left[1]+lwh[1],top_left[2]:top_left[2]+lwh[2]] = voltage
            sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]:top_left[2]+lwh[2]] = inside
            
    @staticmethod
    def planar_mesh_3d(sim, top_left, mesh_axis, dims, spacing, thickness, voltage):
        """
        A mesh starting at <top_left> of size <dims> held at <voltage. The distance between "posts"
        on the mesh in each direction is <spacing> and the posts are <thickness thick in all directions.
        The mesh is oriented along <mesh_axis>.
        """
        top_left = sim.global_unit_to_point(top_left)
        dims = sim.unit_to_point(dims)
        spacing = (max(round(spacing[0] * sim.scale), 1), max(round(spacing[1] * sim.scale), 1))
        thickness = max(round(thickness * sim.scale), 1)
        
        region = sim.V[top_left[0]:top_left[0]+dims[0],top_left[1]:top_left[1]+dims[1],top_left[2]:top_left[2]+dims[2]]
        
        if mesh_axis == 0:
            for i in range(0, (dims[1] // spacing[0]) + 1):
                region[:,(i * spacing[0]):(i * spacing[0])+thickness,:] = voltage
            for j in range(0, (dims[1] // spacing[1]) + 1):
                region[:,:,(j * spacing[1]):(j * spacing[1])+thickness] = voltage
        elif mesh_axis == 1:
            for i in range(0, (dims[0] // spacing[0]) + 1):
                region[(i * spacing[0]):(i * spacing[0])+thickness,:,:] = voltage
            for j in range(0, (dims[2] // spacing[1]) + 1):
                region[:,:,(j * spacing[1]):(j * spacing[1])+thickness] = voltage
        elif mesh_axis == 2:
            for i in range(0, (dims[0] // spacing[0]) + 1):
                region[:,(i * spacing[0]):(i * spacing[0])+thickness,:] = voltage
            for j in range(0, (dims[1] // spacing[1]) + 1):
                region[(j * spacing[1]):(j * spacing[1])+thickness,:,:] = voltage
        
    @staticmethod
    def arbitrary_mask(sim, mask, voltage):
        """
        Holds an arbitrary mask of points <mask> at <voltage>.
        The size of the mask must match sim.V.
        Used with more complex shapes that do not reduce to rectangular selections.
        """
        for point in np.array(np.nonzero(mask)).T:
            sim.V[tuple(point)] = voltage

class EMObjectMasks:
    @staticmethod
    def ring_in_plane(sim, center, radius, half_thickness, axis):
        """
        Create a mask representing a circular ring aligned with a plane.
        The ring is at <center> with <radius>, and the thickness of the ring is 2*<half_thickness>.
        The ring is oriented along <axis>.
        Returns a numpy array of the same size as sim.V with 1s in the locations of the ring.
        """
        mask = np.zeros(sim.V.shape, int)
        
        radius = round(radius * sim.scale)
        ht = max(1, round(half_thickness * sim.scale))
        center = sim.global_unit_to_point(center)
        
        theta_step = ((2.0 / sim.scale)**0.5) / radius
        for i in range(int((2 * np.pi) / theta_step) + 1):
            angle = i * theta_step
            if axis == 0:
                point = np.array([radius * np.cos(angle) + center[1], radius * np.sin(angle) + center[2]]).astype(int)
                mask[center[0],point[0]-ht:point[0]+ht,point[1]-ht:point[1]+ht] = 1
            elif axis == 1:
                point = np.array([radius * np.cos(angle) + center[0], radius * np.sin(angle) + center[2]]).astype(int)
                mask[point[0]-ht:point[0]+ht,center[1],point[1]-ht:point[1]+ht] = 1
            elif axis == 2:
                point = np.array([radius * np.cos(angle) + center[0], radius * np.sin(angle) + center[1]]).astype(int)
                mask[point[0]-ht:point[0]+ht,point[1]-ht:point[1]+ht,center[2]] = 1
        return mask
    
    @staticmethod
    def open_cylinder_in_plane(sim, center, radius, half_thickness, length, axis, direction=1):
        """
        Create a mask representing a circular cylinder aligned with a plane.
        The cylinder is at <center> with <radius>, and the thickness of the ring is 2*<half_thickness>.
        The <length> of the cylinder extends in the sign of <direction> from the <center> along <axis>.
        Returns a numpy array of the same size as sim.V with 1s in the locations of the cylinder.
        """
        mask = np.zeros(sim.V.shape, int)
        
        radius = round(radius * sim.scale)
        ht = max(1, round(half_thickness * sim.scale))
        length = round(length * sim.scale)
        center = sim.global_unit_to_point(center)

        cyl_start = center[axis] if direction == 1 else center[axis] - length
        cyl_end = center[axis] + length if direction == 1 else center[axis]
        
        theta_step = ((2.0 / sim.scale)**0.5) / radius
        for i in range(int((2 * np.pi) / theta_step) + 1):
            angle = i * theta_step
            if axis == 0:
                point = np.array([radius * np.cos(angle) + center[1], radius * np.sin(angle) + center[2]]).astype(int)
                mask[cyl_start:cyl_end,point[0]-ht:point[0]+ht,point[1]-ht:point[1]+ht] = 1
            elif axis == 1:
                point = np.array([radius * np.cos(angle) + center[0], radius * np.sin(angle) + center[2]]).astype(int)
                mask[point[0]-ht:point[0]+ht,cyl_start:cyl_end,point[1]-ht:point[1]+ht] = 1
            elif axis == 2:
                point = np.array([radius * np.cos(angle) + center[0], radius * np.sin(angle) + center[1]]).astype(int)
                mask[point[0]-ht:point[0]+ht,point[1]-ht:point[1]+ht,cyl_start:cyl_end] = 1
        return mask
    
    @staticmethod
    def closed_cylinder_in_plane(sim, center, radius, length, axis, direction=1):
        """
        Create a mask representing a filled circular cylinder aligned with a plane.
        The cylinder is at <center> with <radius>.
        The <length> of the cylinder extends in the sign of <direction> from the <center> along <axis>.
        Returns a numpy array of the same size as sim.V with 1s in the locations of the cylinder.
        """
        mask = np.zeros(sim.V.shape, int)
        
        radius = round(radius * sim.scale)
        length = round(length * sim.scale)
        center = sim.global_unit_to_point(center)

        cyl_start = center[axis] if direction == 1 else center[axis] - length
        cyl_end = center[axis] + length if direction == 1 else center[axis]
        
        for xsec in np.arange(-radius, radius+1, 1):
            hwidth = round(((radius**2) - (xsec**2))**0.5)
            if axis == 0:
                mask[cyl_start:cyl_end,center[1]-hwidth:center[1]+hwidth,center[2]+xsec] = 1
            elif axis == 1:
                mask[center[0]-hwidth:center[0]+hwidth,cyl_start:cyl_end,center[2]+xsec] = 1
            elif axis == 2:
                mask[center[0]-hwidth:center[0]+hwidth,center[1]+xsec,cyl_start:cyl_end] = 1
        
        return mask
            