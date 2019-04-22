from pyscf import gto
from hf_helper import HFHelper
import numpy as np


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

nmo = mol.nao
nocc = mol.nelec[0]
so = slice(0, nocc)
sa = slice(0, nmo)

scfh = HFHelper(mol)


class TestHFHelper(object):
    def test_grad_elec_by_mo(self):
        assert np.allclose(scfh._refimp_grad_elec_by_mo(), scfh.scf_grad.grad_elec())

    def test_grad_elec_by_ao(self):
        assert np.allclose(scfh._refimp_grad_elec(), scfh.scf_grad.grad_elec())

    def test_grad_nuc(self):
        assert(np.allclose(scfh._refimp_grad_nuc(), scfh.scf_grad.grad_nuc()))

    def test_grad(self):
        assert(np.allclose(scfh._refimp_grad(), scfh.get_grad()))

    def test_hess_elec(self):
        assert np.allclose(scfh._refimp_hess_elec(), scfh.scf_hess.hess_elec())
