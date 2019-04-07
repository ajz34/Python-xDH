import sys
sys.path.append('../../')

from hf_helper import HFHelper
from gga_helper import GGAHelper
from ncgga_engine import NCGGAEngine
from numeric_helper import NumericDiff
import numpy as np
from functools import partial
from pyscf import gto, dft, mp, grad, scf
import pickle

np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * 2 / 8])
print = partial(print, flush=True)
np.set_printoptions(8, linewidth=1000, suppress=True)


if __name__ == "__main__":

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

    scfh_ = GGAHelper(mol, "b3lypg", grids)
    nch_ = GGAHelper(mol, "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", grids, init_scf=False)
    ncengine = NCGGAEngine(scfh_, nch_)

    def mol_to_xyg3(mol):
        scfh = GGAHelper(mol, "b3lypg", grids)
        nch = GGAHelper(mol, "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", grids, init_scf=False)
        ncengine = NCGGAEngine(scfh, nch)
        e0 = ncengine.get_E_0()
        mp2_eng = mp.MP2(scfh.scf_eng)
        mp2_eng.kernel()
        return e0 + 0.3211 * mp2_eng.e_corr

    print(mol_to_xyg3(mol))

    print(NumericDiff(mol, mol_to_xyg3).get_numdif())

#    print(ncengine.get_E_1())
#    print("------")
    # print(scfh_.get_grad())
    # print("------")

#    hf_eng = scf.RHF(mol)
#    hf_eng.mo_coeff = scfh_.scf_eng.mo_coeff
#    hf_eng.mo_energy = scfh_.scf_eng.mo_energy
#    hf_eng.mo_occ = scfh_.scf_eng.mo_occ
#    hf_grad = grad.RHF(hf_eng)
#    print(hf_grad.kernel())
#    print(hf_grad.de)
#    print("------")
#
#    mp2_eng = mp.MP2(hf_eng)
#    mp2_eng.kernel()
#    mp2_grad = grad.mp2.Gradients(mp2_eng)
#    mp2_grad.kernel()
#    print(mp2_grad.de)
#    print(0.3211 * (mp2_grad.de - hf_grad.de) + ncengine.get_E_1())
#
#    print("------")
#    scf_eng = dft.RKS(mol)
#    scf_eng.xc = "b3lypg"
#    scf_eng.grids = grids
#    scf_eng.run()
#    scf_grad = grad.rks.Gradients(scf_eng)
#    scf_grad.kernel()
#    print(scf_grad.de)
