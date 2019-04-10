import sys, os
sys.path.append('../../')

MAXCORE = "24"
os.environ["MAXMEM"] = "24"
os.environ["OMP_NUM_THREADS"] = MAXCORE
os.environ["OPENBLAS_NUM_THREADS"] = MAXCORE
os.environ["MKL_NUM_THREADS"] = MAXCORE
os.environ["VECLIB_MAXIMUM_THREADS"] = MAXCORE
os.environ["NUMEXPR_NUM_THREADS"] = MAXCORE


from hf_helper import HFHelper
from gga_helper import GGAHelper
from ncgga_engine import NCGGAEngine
from functools import partial
from pyscf import gto, dft


print = partial(print, flush=True)


if __name__ == "__main__":
    import time

    time0 = time.time()
    mol = gto.Mole()
    mol.atom = """
        C    -1.412680    0.001521    0.009544
        C    -0.703590    0.643016   -1.012670
        C     0.692727    0.642100   -1.025980
        C     1.400490   -0.004232   -0.011071
        C     0.705071   -0.647588    1.013950
        C    -0.691398   -0.642812    1.021170
        H    -2.512700   -0.004404    0.012425
        H    -1.250050    1.165412   -1.815509
        H     1.237330    1.160145   -1.833324
        H     2.503520    0.001674   -0.014122
        H     1.259250   -1.151864    1.823627
        H    -1.227970   -1.162968    1.831961
        """
    mol.basis = "aug-cc-pVTZ"
    mol.verbose = 0
    mol.max_memory = 24000
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

    print("E_2 time             ", time.time() - time0)
    time0 = time.time()
    print(ncengine.E_2)
