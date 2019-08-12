import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class UnitCell:
    def __init__(self, positions=[]):
        self.ATOMS_ = positions
        #self.SCATTER_FACTOR = scattering_amplitude

class SimulationCell:
    def __init__(self, UnitCell, latticedistort, dim="2D", latticeVectors=[], lattice_dimensions=[]):
        self.UC_ = UnitCell
        self.DIM_ = dim
        self.LVEC_ = [np.array(l) for l in latticeVectors]
        self.RVEC_ = self.calculate_reciprocal(latticeVectors)
        self.LDIM_ = lattice_dimensions
        self.LAT_, self.LAT_ATOMS_ = self.build_simulation_volume(latticedistort)

    def calculate_reciprocal(self, latticeVectors):
        reciprocalVectors = []
        if self.DIM_ == "2D":
            vec0, vec1 = latticeVectors
            vec2 = np.cross(vec0, vec1)

            Volume = np.linalg.norm(np.dot(vec0, np.cross(vec1,vec2)))
            rec0 = 2.*np.pi*np.cross(vec1,vec2)/Volume
            rec1 = 2.*np.pi*np.cross(vec2,vec0)/Volume
            rec2 = 2.*np.pi*np.cross(vec0,vec1)/Volume
            reciprocalVectors = [rec0, rec1, rec2]

        elif self.DIM_ == "3D":
            vec0, vec1, vec2 = latticeVectors

            Volume = np.linalg.norm(np.dot(vec0, np.cross(vec1,vec2)))
            rec0 = 2.*np.pi*np.cross(vec1,vec2)/Volume
            rec1 = 2.*np.pi*np.cross(vec2,vec0)/Volume
            rec2 = 2.*np.pi*np.cross(vec0,vec1)/Volume
            reciprocalVectors = [rec0, rec1, rec2]
        
        return reciprocalVectors

    def build_simulation_volume(self, func):
        atom_positions = []
        lattice = []
        M,N = self.LDIM_
        for i in range(-int(np.floor(M/2)), int(np.ceil(M/2))):
            for j in range(-int(np.floor(N/2)), int(np.ceil(N/2))):
                lattice.append(i*self.LVEC_[0] + j*self.LVEC_[1])
                for pos in self.UC_.ATOMS_:
                    position = np.array(pos) + i*self.LVEC_[0] + j*self.LVEC_[1]
                    atom_positions.append(func(position))
        return lattice, atom_positions

    def export_structure(self, filename):
        f_ = open(filename, 'w+')
        f_.write('# ATOMS\n')
        for at in self.LAT_ATOMS_:
            f_.write('%f, %f, %f\n' %(at[0], at[1], at[2]))
        f_.write('# VECTORS\n')
        for vec, dim in zip(self.LVEC_, self.LDIM_):
            f_.write('%f, %f, %f\n' % (dim*vec[0], dim*vec[1], dim*vec[2]))
        f_.write('# LATTICE DIMENSIONS\n')
        f_.write('1, 1')
        f_.close()


class LEED():
    def __init__(self, SimulationCell):
        self.SIMCELL_ = SimulationCell
        
    def Intensity(self, dk, distortion = True):
        if distortion:
            result = self.Intensity_distortion(dk)
        else:
            result = self.Intensity_no_distortion(dk)
        return result
        
    def Intensity_no_distortion(self, dk):
        #dk = np.array([dkx, dky, dkz])
        lattice = np.abs(np.sum(np.exp(-1j*np.dot(self.SIMCELL_.LAT_, dk)), axis=0))**2
        structure = np.abs(np.sum(np.exp(-1j*np.dot(self.SIMCELL_.UC_.ATOMS_, dk)), axis=0))**2
        return lattice*structure

    def Intensity_distortion(self, dk):
        #dk = np.array([dkx, dky, dkz])
        #lattice = np.abs(np.sum(np.exp(-1j*np.apply_along_axis(np.dot, 1, self.SIMCELL_.LAT_ATOMS_, dk))))**2
        lattice = np.abs(np.sum(np.exp(-1j*np.dot(self.SIMCELL_.LAT_ATOMS_, dk)), axis=0))**2
        return lattice

def latticedistortion(pos, center=[0.,0.], amplitudes=[0,0,0.0]):
    posX, posY, posZ = pos
    #return pos
    return np.array([posX + (posX/np.sqrt(posX**2 + posY**2+1e-6))* amplitudes[0]*np.cos(1.*np.sqrt((posX-center[0])**2 + (posY-center[1])**2)),
     posY+(posY/np.sqrt(posX**2 + posY**2+1e-6))*amplitudes[1]*np.cos(1.*np.sqrt((posX-center[0])**2 + (posY-center[1])**2)), 
     posZ + amplitudes[2]*np.cos(1.*np.sqrt((posX-center[0])**2 + (posY-center[1])**2))])

def load_structure(filelocation):
    f = open(filelocation, 'r')
    lines = f.readlines()
    f.close()
    last_tag = ''
    atoms = []
    vectors = []
    dimensions = []
    for line in lines:
        if last_tag == '# ATOMS':
            if line[0] != '#':
                atoms.append([float(l) for l in line.split(',')])
        elif last_tag == '# VECTORS':
            if line[0] != '#':
                vectors.append([float(l) for l in line.split(',')])
        elif last_tag == '# LATTICE DIMENSIONS':
            if line[0] != '#':
                dimensions = [int(l) for l in line.split(',')]
        if line[0] == '#':
            last_tag = line.strip()
    return atoms, vectors, dimensions

atoms, vectors, dimensions = load_structure('./test.txt')

Uc = UnitCell(positions=atoms)
Sc = SimulationCell(Uc, latticedistortion,latticeVectors=vectors, lattice_dimensions=[10,10])
#Sc.export_structure('test2.txt')
Ld = LEED(Sc)

kxmin, kxmax, Nx = -3, 3, 200
kymin, kymax, Ny = -3, 3, 200
kxcoords = np.linspace(kxmin,kxmax,Nx)
kycoords = np.linspace(kymin,kymax,Ny)

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for at in Sc.LAT_ATOMS_:
        ax.scatter(at[0], at[1], at[2])
    plt.show()

plt.figure()
t0 = time.time()
kvals = np.array([[[kx, ky, 4.2 - np.sqrt(kx**2 + ky**2)] for ky in kycoords] for kx in kxcoords]).transpose(2,0,1).reshape(3,-1)
ptrn = Ld.Intensity(kvals, distortion=False)
t1 = time.time()
print(t1 - t0)
plt.imshow(np.nan_to_num(ptrn.reshape(Nx,Ny)), extent=[kxmin, kxmax, kymin, kymax])

plt.show()