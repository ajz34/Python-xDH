import numpy as np
from pyscf import gto, dft
from functools import partial

import sys, gc, os
sys.path.append('../../')
os.environ["LOGLEVEL"] = "2"
import pickle
from gga_helper import GGAHelper
from utilities import timing_level

np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * 10 / 8])
np.einsum_path = partial(np.einsum_path, optimize=["greedy", 1024 ** 3 * 10 / 8])
np.set_printoptions(5, linewidth=120, suppress=True)


@timing_level(0)
def main():
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
    grids.atom_grid = (75, 302)
    grids.becke_scheme = dft.gen_grid.stratmann
    grids.build()

    scfh = GGAHelper(mol, "b3lypg", grids)
    # scfh.get_kerh(deriv=3)

    with open('num_C_U1.dat', 'rb') as f:
        pickle_load = pickle.load(f)
    C_diff = pickle_load["C_diff"]
    U_1_num = pickle_load["U_1_num"]

    dif1 = U_1_num - scfh.U_1
    print("U_1    Minimum Deviation: {:10.6e}".format(abs(dif1.min())))
    print("U_1    Maximum Deviation: {:10.6e}".format(abs(dif1.max())))

    with open('num_U1_U2.dat', 'rb') as f:
        pickle_load = pickle.load(f)
    U1vo_diff = pickle_load["U_1_vo_diff"]
    U2vo_num = pickle_load["U_2_vo_num"]

    dif2 = U2vo_num - scfh.U_2_vo
    print("U_2_vo Minimum Deviation: {:10.6e}".format(abs(dif2.min())))
    print("U_2_vo Maximum Deviation: {:10.6e}".format(abs(dif2.max())))

    print("----------------------------------------------------------")
    print("U_1    Minimum Deviation: {:10.6e}".format(abs(dif1.min())))
    print("U_1    Maximum Deviation: {:10.6e}".format(abs(dif1.max())))
    print("U_2_vo Minimum Deviation: {:10.6e}".format(abs(dif2.min())))
    print("U_2_vo Maximum Deviation: {:10.6e}".format(abs(dif2.max())))


if __name__ == "__main__":
    main()
