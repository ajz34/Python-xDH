import os

MAXCORE = "8"
os.environ["MAXMEM"] = "12"
os.environ["OMP_NUM_THREADS"] = MAXCORE
os.environ["OPENBLAS_NUM_THREADS"] = MAXCORE
os.environ["MKL_NUM_THREADS"] = MAXCORE
os.environ["VECLIB_MAXIMUM_THREADS"] = MAXCORE
os.environ["NUMEXPR_NUM_THREADS"] = MAXCORE
os.environ["LOGLEVEL"] = "2"

import numpy as np
from pyscf import gto, dft
from functools import partial
import pickle
from utilities.numeric_helper import NumericDiff
from hessian.gga_helper import GGAHelper

np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * 2 / 8])
np.einsum_path = partial(np.einsum_path, optimize=["greedy", 1024 ** 3 * 2 / 8])
np.set_printoptions(5, linewidth=120, suppress=True)


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
    mol.max_memory = 12000

    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.becke_scheme = dft.gen_grid.stratmann
    grids.prune = None
    grids.build()

    nmo = nao = mol.nao
    natm = mol.natm
    nocc = mol.nelec[0]
    nvir = nmo - nocc
    so = slice(0, nocc)
    sv = slice(nocc, nmo)
    sa = slice(0, nmo)

    scfh = GGAHelper(mol, "b3lypg", grids)
    C = scfh.C
    U_1 = scfh.U_1

    C_diff = NumericDiff(mol, lambda mol: GGAHelper(mol, "b3lypg", grids).C, interval=1e-4).get_numdif()
    U_1_num = np.linalg.inv(C) @ C_diff

    with open('num_C_U1.dat', 'wb') as f:
        pickle.dump({"C_diff": C_diff, "U_1_num": U_1_num}, f, pickle.HIGHEST_PROTOCOL)

    U_1_vo_diff = NumericDiff(mol, lambda mol: GGAHelper(mol, "b3lypg", grids).U_1_vo, interval=1e-4, deriv=2, symm=False).get_numdif()
    U_2_vo_num = U_1_vo_diff + np.einsum("Bsam, Atmi -> ABtsai", U_1[:, :, sv, :], U_1[:, :, :, so])

    with open('num_U1_U2.dat', 'wb') as f:
        pickle.dump({"U_1_vo_diff": U_1_vo_diff, "U_2_vo_num": U_2_vo_num}, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
