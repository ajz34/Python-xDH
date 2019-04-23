from gga_helper import GGAHelper
from ncgga_engine import NCGGAEngine
from numeric_helper import NumericDiff
import numpy as np
from pyscf import gto, dft
import pickle
from utilities import timing_level

np.set_printoptions(8, linewidth=1000, suppress=True)


@timing_level(0)
def get_numeric_0(mol, grids):
    scfh = GGAHelper(mol, "b3lypg", grids, init_scf=True)
    nch = GGAHelper(mol, "pbe0", grids, init_scf=False)
    ncengine = NCGGAEngine(scfh, nch)
    return ncengine.get_E_0()


@timing_level(0)
def get_numeric_1(mol, grids):
    scfh = GGAHelper(mol, "b3lypg", grids, init_scf=True)
    nch = GGAHelper(mol, "pbe0", grids, init_scf=False)
    ncengine = NCGGAEngine(scfh, nch)
    return ncengine.get_E_1()


@timing_level(0)
def get_nc(mol, grids):

    scfh = GGAHelper(mol, "b3lypg", grids, init_scf=True)
    nch = GGAHelper(mol, "pbe0", grids, init_scf=False)
    ncengine = NCGGAEngine(scfh, nch)

    d = {
        "E_0": ncengine.get_E_0(),
        "E_1": ncengine.get_E_1(),
        "E_2": ncengine.get_E_2(),
    }
    with open("H2O2-bak.dat", "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    return d


@timing_level(0)
def get_numeric(mol, grids):

    d = {
        "E_0_diff": NumericDiff(mol, lambda m: get_numeric_0(m, grids), interval=1e-4).get_numdif(),
        "E_1_diff": NumericDiff(mol, lambda m: get_numeric_1(m, grids), interval=1e-4, deriv=2).get_numdif(),
    }
    with open("H2O2-bak-numeric.dat", "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    return d


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

    d_nc = get_nc(mol, grids)
    d_num = get_numeric(mol, grids)

    print("E_0 diff max: ", (d_nc["E_1"] - d_num["E_0_diff"]).max())
    print("E_0 diff min: ", (d_nc["E_1"] - d_num["E_0_diff"]).min())
    print("E_1 diff max: ", (d_nc["E_2"] - d_num["E_1_diff"]).max())
    print("E_1 diff min: ", (d_nc["E_2"] - d_num["E_1_diff"]).min())


if __name__ == "__main__":
    main()
