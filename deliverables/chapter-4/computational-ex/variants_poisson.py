import firedrake as fd
from firedrake.output import VTKFile 
from firedrake.__future__ import interpolate 

import gmsh
import numpy as np

# User defined data
bottom_wall = 0
right_wall  = 1
top_wall    = 2
left_wall   = 3
hole1_wall  = 4
hole2_wall  = 5
hole3_wall  = 6

inclusion_marker = 3
background_marker = 2

ninclusions = 3
Lx = 2.0
Ly = 2.0
R1 = 0.25
R2 = 0.15
R3 = 0.25

gdim = 2

#--------------------------------------------------------------------
#--- Preprocess: Mesh generation, boundary and region identification

def GenerateMesh():

    gmsh.initialize()

    # We create one rectangle and the circular inclusion
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    hole1 = gmsh.model.occ.addDisk(0.5, 1.0, 0, R1, R1)
    hole2 = gmsh.model.occ.addDisk(1.0, 1.5, 0, R2, R2)
    hole3 = gmsh.model.occ.addDisk(1.5, 1.0, 0, R3, R3)
    gmsh.model.occ.synchronize()
    all_holes = [(2, hole1)]
    all_holes.extend([(2, hole2)])
    all_holes.extend([(2, hole3)])
    whole_domain = gmsh.model.occ.cut([(gdim, rectangle)], all_holes)
    gmsh.model.occ.synchronize()
    background_surfaces = []
    for domain in whole_domain[0]:
        gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=background_marker)
        background_surfaces.append(domain)
            
    # Tag the the different boundaries
    left = []
    right = []
    top = []
    bottom = []
    hole1,hole2,hole3 = [], [], []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0.0):
            #print('L', line, com)
            left.append(line[1])
        if np.isclose(com[0], Lx):
            #print('R', line, com)
            right.append(line[1])
        if np.isclose(com[1], 0.0):
            #print('B', line, com)
            bottom.append(line[1])
        if np.isclose(com[1], Ly):
            #print('T', line, com)
            top.append(line[1])
        if np.isclose(np.linalg.norm(com), np.sqrt((0.5)**2 + (1.0)**2)):
            #print('H1', line, com)
            hole1.append(line[1])
        elif np.isclose(np.linalg.norm(com), np.sqrt((1.0)**2 + (1.5)**2)) and np.isclose(com[1], 1.5):
            #print('H2', line, com)
            hole2.append(line[1])
        elif np.isclose(np.linalg.norm(com), np.sqrt((1.5)**2 + (1.0)**2)):
            #print('H3', line, com)
            hole3.append(line[1])
                
    gmsh.model.addPhysicalGroup(1, left, left_wall)
    gmsh.model.addPhysicalGroup(1, right, right_wall)
    gmsh.model.addPhysicalGroup(1, top, top_wall)
    gmsh.model.addPhysicalGroup(1, bottom, bottom_wall)
    gmsh.model.addPhysicalGroup(1, hole1, hole1_wall)
    gmsh.model.addPhysicalGroup(1, hole2, hole2_wall)
    gmsh.model.addPhysicalGroup(1, hole3, hole3_wall)
    gmsh.model.occ.synchronize()

    if(True):
        r = 0.01
        res_min = r
        res_max = 4 * r
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", hole1+hole2+hole3)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(2, "LcMax", res_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 2*r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 4*r)
        # We take the minimum of the two fields as the mesh size
        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        # Generate mesh
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
        gmsh.model.mesh.generate(2)
        
    gmsh.write("mesh.msh")
    gmsh.finalize()
    mesh = fd.Mesh("mesh.msh")

    return mesh
#--------------------------------------------------------------------

mesh = GenerateMesh()

# Material parameters and boundary values
mu = fd.Constant(1.0)
gamma = fd.Constant(0.0)
source = fd.Constant(0.0)
uleft = 100.0
uright = 1.0
T1 = ...
T2 = ...
T3 = ...
ht = fd.Constant(0.0)
hb = fd.Constant(0.0)

Vh = fd.FunctionSpace(mesh, "CG", degree=1)

# Set Dirchlet values
bcs = []
bcs.append(fd.DirichletBC(Vh, fd.Constant(uleft), left_wall))
bcs.append(fd.DirichletBC(Vh, fd.Constant(uright), right_wall))
bcs.append(fd.DirichletBC(Vh, fd.Constant(T1), hole1_wall))
...
...


# Variational formulation
x = fd.SpatialCoordinate(mesh) 
u, v = fd.TrialFunction(Vh), fd.TestFunction(Vh)

a = fd.inner(mu*fd.grad(u), fd.grad(v))*fd.dx + fd.inner(gamma*u,v)*(fd.ds(0) + fd.ds(2))
L = source * v * fd.dx + ...

# Solve the problem
uh = fd.Function(Vh, name='Temperature')
opts={"ksp_type": "preonly", "pc_type": "lu"}
fd.solve(a==L, uh, bcs=bcs, solver_parameters=opts)
VTKFile("temperature.pvd").write(uh)