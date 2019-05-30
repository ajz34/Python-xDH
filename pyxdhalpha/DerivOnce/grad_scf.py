import numpy as np
from functools import partial
import os

from pyscf import grad

from pyxdhalpha.DerivOnce.deriv_once import DerivOnce, DerivOnceNCDFT
from pyxdhalpha.Utilities import GridIterator, KernelHelper

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradSCF(DerivOnce):

    def __init__(self, scf_eng, rotation=True, grdit_memory=2000):
        super(GradSCF, self).__init__(scf_eng, rotation, grdit_memory)

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError("This is still under construction...")

    def _get_H_1_ao(self):
        return np.array([self.scf_grad.hcore_generator()(A) for A in range(self.natm)]).reshape(-1, self.nao, self.nao)

    def _get_F_1_ao(self):
        return np.array(self.scf_hess.make_h1(self.C, self.mo_occ)).reshape(-1, self.nao, self.nao)

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
    
    def _get_E_1(self):
        cx, xc = self.cx, self.xc
        so, sv = self.so, self.sv
        mol, natm = self.mol, self.natm
        D = self.D
        H_1_ao = self.H_1_ao
        eri1_ao = self.eri1_ao
        S_1_mo = self.S_1_mo
        F_0_mo = self.F_0_mo
        grids = self.grids
        grdit_memory = self.grdit_memory

        grad_total = (
            + np.einsum("Auv, uv -> A", H_1_ao, D)
            + 0.5 * np.einsum("Auvkl, uv, kl -> A", eri1_ao, D, D)
            - 0.25 * cx * np.einsum("Aukvl, uv, kl -> A", eri1_ao, D, D)
            - 2 * np.einsum("Aij, ij -> A", S_1_mo[:, so, so], F_0_mo[so, so])
            + grad.rhf.grad_nuc(mol).reshape(-1)
        )

        # GGA part contiribution
        if self.xc_type == "GGA":
            grdit = GridIterator(mol, grids, D, deriv=2, memory=grdit_memory)
            for grdh in grdit:
                kerh = KernelHelper(grdh, xc)
                grad_total += (
                    + np.einsum("g, Atg -> At", kerh.fr, grdh.A_rho_1)
                    + 2 * np.einsum("g, rg, Atrg -> At", kerh.fg, grdh.rho_1, grdh.A_rho_2)
                ).reshape(-1)

        return grad_total.reshape(natm, 3)


class GradNCDFT(DerivOnceNCDFT, GradSCF):

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError("This is still under construction...")

    def _get_E_1(self):
        raise NotImplementedError("This is still under construction...")


class Test_GradSCF:

    def test_HF_grad(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2()
        gsh = GradSCF(H2O2.hf_eng)
        hf_grad = gsh.scf_grad

        assert(np.allclose(
            gsh.E_1, hf_grad.grad(),
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-HF-freq.fchk"))

        assert(np.allclose(
            gsh.E_1, formchk.grad(),
            atol=1e-6, rtol=1e-4
        ))

    def test_B3LYP_grad(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2()
        gsh = GradSCF(H2O2.gga_eng)
        gga_grad = gsh.scf_grad

        assert(np.allclose(
            gsh.E_1, gga_grad.grad(),
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-B3LYP-freq.fchk"))

        # TODO: This is a weaker compare! Try to modulize that someday.
        assert(np.allclose(
            gsh.E_1, formchk.grad(),
            atol=1e-5, rtol=1e-4
        ))
