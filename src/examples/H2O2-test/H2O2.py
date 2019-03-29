import sys
sys.path.append('../../')

from hf_helper import HFHelper
from gga_helper import GGAHelper
from ncgga_engine import NCGGAEngine
import numpy as np
from functools import partial
from pyscf import gto, dft

np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * 2 / 8])
print = partial(print, flush=True)
np.set_printoptions(8, linewidth=1000, suppress=True)


mol = gto.Mole()
mol.atom = """
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.5  0.0  0.0
H  0.0  0.7  1.5
"""
mol.basis = "6-31G"
mol.verbose = 0
mol.build()

grids = dft.gen_grid.Grids(mol)
grids.atom_grid = (99, 590)
grids.becke_scheme = dft.gen_grid.stratmann
grids.build()

scfh_ = HFHelper(mol)
nch_ = GGAHelper(mol, "b3lypg", grids, init_scf=False)
ncengine = NCGGAEngine(scfh_, nch_)

print(ncengine.get_E_0())
print(ncengine.get_E_1())
print(ncengine.get_E_2())

