import gmsh
import numpy as np

def CreateMeshBody():

    gmsh.initialize()

    lc = 0.05
    
    Db   = 0.4
    Hb   = 0.4
    Hp   = 6*Hb
    R    = 3*Hb
    TT   = np.sqrt(R*R - 4*Hb*Hb)
    
    gmsh.model.occ.addPoint(0, 0, 0, lc, 1)
    gmsh.model.occ.addPoint(Db, 0, 0, lc, 2)
    gmsh.model.occ.addPoint(Db, Hb, 0, 0.5*lc, 3)
    gmsh.model.occ.addPoint(TT+Db, 3*Hb, 0, lc, 4)
    gmsh.model.occ.addPoint(Db, 5*Hb, 0, lc, 5)
    gmsh.model.occ.addPoint(Db, 6*Hb, 0, 0.5*lc, 6)
    gmsh.model.occ.addPoint(0, 6*Hb, 0, lc, 7)
    gmsh.model.occ.addPoint(0, 3*Hb, 0, 0.1*lc, 8)
    gmsh.model.occ.addPoint(TT+Db-R, 3*Hb, 0, 0.1*lc, 9)
    
    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)

    gmsh.model.occ.addCircleArc(3, 4, 9, 3)
    gmsh.model.occ.addCircleArc(9, 4, 5, 4)
    
    gmsh.model.occ.addLine(5, 6, 5)
    gmsh.model.occ.addLine(6, 7, 6)
    gmsh.model.occ.addLine(7, 8, 7)
    gmsh.model.occ.addLine(8, 1, 8)
    gmsh.model.occ.synchronize()

    bottomwall, topwall = [], []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[1], 0):
            bottomwall.append(line[1])
        if np.isclose(com[1], 6*Hb):
            topwall.append(line[1])
    
    gmsh.model.addPhysicalGroup(1, bottomwall, 200)
    gmsh.model.addPhysicalGroup(1, topwall, 300)
    
    gmsh.model.occ.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
    gmsh.model.occ.addPlaneSurface([1], 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7, 8], 101)
    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface") 
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    gmsh.write("corpo.msh")
    gmsh.finalize()

    tags = {'bottom': 200, 'top': 300}

    mesh = fd.Mesh("corpo.msh")

    return mesh, tags