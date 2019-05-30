import numpy as np
from functools import partial
import os

from pyscf import gto, scf, dft, grad, hessian, lib
import pyscf.dft.numint
import pyscf.scf.cphf

from pyxdhalpha.Utilities import timing, GridIterator
from pyxdhalpha.Utilities.grid_helper import KernelHelper
from pyxdhalpha.DerivOnce.deriv_once import DerivOnce

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradSCF(DerivOnce):

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError("This is still under construction...")

    def _get_H_1_ao(self):
        return np.array([self.scf_grad.hcore_generator()(A) for A in range(self.natm)]).reshape(-1, self.nao, self.nao)

    def _get_F_1_ao(self):
        return self.scf_hess.make_h1(self.C, self.mo_occ).reshape(-1, self.nao, self.nao)

    def _get_S_1_ao(self):
        int1e_ipovlp = self.mol.intor("int1e_ipovlp")

        def get_S_S_ao(A):
            ao_matrix = np.zeros((3, self.nao, self.nao))
            sA = self.mol_slice(A)
            ao_matrix[:, sA] = -int1e_ipovlp[:, sA]
            return ao_matrix + ao_matrix.swapaxes(1, 2)

        S_1_ao = np.array([get_S_S_ao(A) for A in range(self.natm)]).reshape(-1, self.nao, self.nao)
        return S_1_ao

    def _get_eri1_ao(self):
        nao = self.nao
        natm = self.natm
        int2e_ip1 = self.mol.intor("int2e_ip1")
        eri1_ao = np.zeros((natm, 3, nao, nao, nao, nao))
        for A in range(natm):
            sA = self.mol_slice(A)
            eri1_ao[A, :, sA, :, :, :] -= int2e_ip1[:, sA]
            eri1_ao[A, :, :, sA, :, :] -= int2e_ip1[:, sA].transpose(0, 2, 1, 3, 4)
            eri1_ao[A, :, :, :, sA, :] -= int2e_ip1[:, sA].transpose(0, 3, 4, 1, 2)
            eri1_ao[A, :, :, :, :, sA] -= int2e_ip1[:, sA].transpose(0, 3, 4, 2, 1)
        return eri1_ao.reshape(-1, self.nao, self.nao, self.nao, self.nao)


class Test_GradSCF:

    def test_HF_grad(self):
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        H2O2 = Mol_H2O2()
        gsh = GradSCF(H2O2.hf_eng)
        hf_grad = gsh.scf_grad
        D = gsh.D
        H_1_ao = gsh.H_1_ao
        eri1_ao = gsh.eri1_ao
        S_1_ao = gsh.S_1_ao
        Co = gsh.Co
        eo = gsh.eo

        assert(np.allclose(
            + np.einsum("Auv, uv -> A", H_1_ao, D)
            + 0.5 * np.einsum("Auvkl, uv, kl -> A", eri1_ao, D, D)
            - 0.25 * np.einsum("Aukvl, uv, kl -> A", eri1_ao, D, D)
            - 2 * np.einsum("Auv, ui, i, vi -> A", S_1_ao, Co, eo, Co),
            hf_grad.grad_elec().reshape(-1)
        ))
