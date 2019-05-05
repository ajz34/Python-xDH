import sys
sys.path.append('../../')

from hessian.hf_helper import HFHelper
from hessian.gga_helper import GGAHelper
from hessian.ncgga_engine import NCGGAEngine
from utilities.numeric_helper import NumericDiff
import numpy as np
from functools import partial
from pyscf import gto, dft

np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * 2 / 8])
print = partial(print, flush=True)
np.set_printoptions(8, linewidth=1000, suppress=True)


if __name__ == "__main__":

    import time

    time0 = time.time()
    mol = gto.Mole()
    mol.atom = """
        C                  0.00000000    0.00000000    0.00000000
        H                 -0.00000000    0.00000000    1.06999996
        H                 -0.00000000   -1.00880563   -0.35666665
        H                 -0.87365131    0.50440282   -0.35666665
        H                  0.87365131    0.50440282   -0.35666665
        """
    mol.basis = "6-31G"
    mol.verbose = 0
    mol.max_memory = 48000
    mol.build()

    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.becke_scheme = dft.gen_grid.stratmann
    grids.build()

    scfh = HFHelper(mol)
    nch = GGAHelper(mol, "b3lypg", grids, init_scf=False)
    ncengine = NCGGAEngine(scfh, nch)

    print("Initialization time: ", time.time() - time0)
    time0 = time.time()

    ncengine.get_E_0()
    print("E_0 time             ", time.time() - time0)
    time0 = time.time()
    print(ncengine.E_0)

    ncengine.get_E_1()
    print("E_1 time             ", time.time() - time0)
    time0 = time.time()
    print(ncengine.E_1)

    ncengine.get_E_SS()
    print("E_SS time            ", time.time() - time0)
    time0 = time.time()

    ncengine.get_E_SU()
    print("E_SU time            ", time.time() - time0)
    time0 = time.time()

    ncengine.get_E_UU()
    print("E_UU time            ", time.time() - time0)
    time0 = time.time()

    print(ncengine.get_E_2())
    print("E_2 time             ", time.time() - time0)
    time0 = time.time()
    print(ncengine.E_2)

    def mol_to_E_0(mol):
        scfh_ = HFHelper(mol)
        nch_ = GGAHelper(mol, "b3lypg", grids, init_scf=False)
        ncengine_ = NCGGAEngine(scfh_, nch_)
        return ncengine_.get_E_0()

    def mol_to_E_1(mol):
        scfh_ = HFHelper(mol)
        nch_ = GGAHelper(mol, "b3lypg", grids, init_scf=False)
        ncengine_ = NCGGAEngine(scfh_, nch_)
        return ncengine_.get_E_1()

    E_0_diff = NumericDiff(mol, mol_to_E_0, interval=1e-4).get_numdif()
    print("E_0_diff             ", time.time() - time0)
    time0 = time.time()
    print(abs(ncengine.E_1 - E_0_diff).max())

    E_1_diff = NumericDiff(mol, mol_to_E_1, interval=1e-4, deriv=2).get_numdif()
    print("E_1_diff             ", time.time() - time0)
    time0 = time.time()
    print(abs(ncengine.E_2 - E_1_diff).max())

    
