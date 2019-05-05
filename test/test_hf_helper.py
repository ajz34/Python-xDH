from pyscf import gto
from hessian import HFHelper
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

nao = nmo = mol.nao
nocc = mol.nelec[0]
so = slice(0, nocc)
sa = slice(0, nmo)

scfh = HFHelper(mol)

X = np.random.random((nao, nao))
X += X.T


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

    def test_hess_nuc(self):
        assert np.allclose(scfh._refimp_hess_nuc(), scfh.scf_hess.hess_nuc())

    def test_hess(self):
        hess = scfh._refimp_hess()
        # Definition
        assert np.allclose(hess, scfh.get_hess())
        # Symmetric
        assert np.allclose(hess, hess.transpose((1, 0, 3, 2)))

    def test_H_0_ao(self):
        H_0_ao = scfh._refimp_H_0_ao()
        # Definition
        assert np.allclose(H_0_ao, scfh.H_0_ao)
        # Symmetric
        assert np.allclose(H_0_ao, H_0_ao.T)

    def test_J_0_ao(self):
        J_0_ao = scfh._refimp_J_0_ao(X)
        # Definition
        assert np.allclose(J_0_ao, scfh.scf_eng.get_j(dm=X))
        # Symmetric
        assert np.allclose(J_0_ao, J_0_ao.T)

    def test_K_0_ao(self):
        K_0_ao = scfh._refimp_K_0_ao(X)
        # Definition
        assert np.allclose(K_0_ao, scfh.scf_eng.get_k(dm=X))
        # Symmetric
        assert np.allclose(K_0_ao, K_0_ao.T)

    def test_F_0_ao(self):
        F_0_ao_X = scfh._refimp_F_0_ao(X)
        F_0_ao = scfh._refimp_F_0_ao(scfh.D)
        # Definition
        assert np.allclose(F_0_ao_X, scfh.scf_eng.get_fock(dm=X))
        # Symmetric
        assert np.allclose(F_0_ao_X, F_0_ao_X.T)
        # Fock matrix with SCF density
        assert np.allclose(F_0_ao, scfh.F_0_ao)
