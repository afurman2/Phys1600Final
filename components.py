def make_enforcer(*lambdas):
    return lambda sim: [f(sim) for f in lambdas]

def enf(func, *args, **kwargs):
    return lambda sim: func(sim, *args, **kwargs)

class EMObjects:
    @staticmethod
    def point_charge(sim, location, voltage):
        sim.V[sim.global_unit_to_point(location)] = voltage
        
    @staticmethod
    def outer_edge_3d(sim, voltage):
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
        sim.V[0,:,:] = voltage
        sim.V[sim.point_space_size[0]-1,:,:] = voltage
        
        sim.V[:,0,:] = voltage
        sim.V[:,sim.point_space_size[1]-1,:] = voltage
        
        sim.V[:,:,0] = voltage
        sim.V[:,:,sim.point_space_size[2]-1] = voltage
        
    @staticmethod
    def rectangular_prism_solid(sim, top_left, lwh, voltage):
        top_left = sim.global_unit_to_point(top_left)
        lwh = sim.unit_to_point(lwh)
        sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]:top_left[1]+lwh[1],top_left[2]:top_left[2]+lwh[2]] = voltage
        
    @staticmethod
    def rectangular_prism_hollow(sim, top_left, lwh, thickness, voltage):
        top_left = sim.global_unit_to_point(top_left)
        lwh = sim.unit_to_point(lwh)
        th = sim.unit_to_point((thickness,))[0]
        
        inside = sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]+th:top_left[2]+lwh[2]-th].copy()
        sim.V[top_left[0]:top_left[0]+lwh[0],top_left[1]:top_left[1]+lwh[1],top_left[2]:top_left[2]+lwh[2]] = voltage
        sim.V[top_left[0]+th:top_left[0]+lwh[0]-th,top_left[1]+th:top_left[1]+lwh[1]-th,top_left[2]+th:top_left[2]+lwh[2]-th] = inside
        
    @staticmethod
    def rectangular_prism_hollow_nocap(sim, top_left, lwh, thickness, cap_axis, voltage):
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
            