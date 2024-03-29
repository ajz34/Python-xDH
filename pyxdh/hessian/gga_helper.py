from pyscf import gto, scf, grad, hessian, dft
import pyscf.grad.rks
import pyscf.hessian.rks
import pyscf.dft.numint
import numpy as np
from functools import partial
import os
import warnings
import gc

from hessian.hf_helper import HFHelper
from utilities import GridHelper, GridIterator, KernelHelper, timing, gccollect


MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GGAHelper(HFHelper):

    def __init__(self, mol, xc, grids, init_scf=True, grdit_memory=2000):
        self.xc = xc  # type: str
        self.grids = grids  # type: dft.gen_grid.Grids
        self.grdit_memory = grdit_memory
        self.scf_eng = None  # type: dft.rks.RKS
        self.scf_grad = None  # type: pyscf.grad.rks.Gradient
        self.scf_hess = None  # type: hessian.rks.Hessian
        self.cx = None
        self.grdh = None  # type: GridHelper
        self.kerh = None  # type: KernelHelper
        # Internal variable
        self._F_2_ao_GGAcontrib = None
        super(GGAHelper, self).__init__(mol, init_scf=init_scf)
        return

    def initialization_pyscf(self):
        self.scf_eng = scf.RKS(self.mol)
        self.scf_eng.conv_tol = 1e-11
        self.scf_eng.conv_tol_grad = 1e-9
        self.scf_eng.max_cycle = 100
        self.scf_eng.xc = self.xc
        self.scf_eng.grids = self.grids
        if self.init_scf:
            self.eng = self.scf_eng.kernel()
            if not self.scf_eng.converged:
                warnings.warn("SCF not converged!")
        self.scf_grad = grad.rks.Gradients(self.scf_eng)
        self.scf_hess = hessian.rks.Hessian(self.scf_eng)
        self.cx = dft.numint.NumInt().hybrid_coeff(self.xc)
        return

    def get_grdh(self):
        self.grdh = GridHelper(self.mol, self.grids, self.D)
        return self.grdh

    def get_kerh(self, deriv=2):
        if self.grdh is None:
            self.get_grdh()
        self.kerh = KernelHelper(self.grdh, self.xc, deriv=deriv)
        return self.kerh

    def Ax0_Core_deprecate(self, si, sj, sk, sl, reshape=True):
        C = self.C
        if self.kerh is None:
            self.get_kerh()
        grdh = self.grdh
        kerh = self.kerh

        def fx(mo1_):
            mo1 = mo1_.copy()  # type: np.ndarray
            shape1 = list(mo1.shape)
            mo1.shape = (-1, shape1[-2], shape1[-1])
            dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.transpose(0, 2, 1)
            r = np.empty((mo1.shape[0], si.stop - si.start, sj.stop - sj.start))
            for idx, dmX in enumerate(dm1):
                tmp_K = np.einsum("kl, gl -> gk", dmX, grdh.ao_0)
                rho_X_0 = np.einsum("gk, gk -> g", grdh.ao_0, tmp_K)
                rho_X_1 = 2 * np.einsum("rgk, gk -> rg", grdh.ao_1, tmp_K)
                gamma_XD = np.einsum("rg, rg -> g", rho_X_1, grdh.rho_1)
                tmp_M = np.empty((4, grdh.ngrid))
                tmp_M[0] = (
                    np.einsum("g, g -> g", rho_X_0, kerh.frr)
                    + 2 * np.einsum("g, g -> g", gamma_XD, kerh.frg)
                )
                tmp_M[1:4] = (
                    + 4 * np.einsum("g, g, rg -> rg", rho_X_0, kerh.frg, grdh.rho_1)
                    + 8 * np.einsum("g, g, rg -> rg", gamma_XD, kerh.fgg, grdh.rho_1)
                    + 4 * np.einsum("rg, g -> rg", rho_X_1, kerh.fg)
                )
                ax_ao = (
                    + 1 * self.scf_eng.get_j(dm=dmX)
                    - 0.5 * self.cx * self.scf_eng.get_k(dm=dmX)
                    + np.einsum("rg, rgu, gv -> uv", tmp_M, grdh.ao[:4], grdh.ao_0)
                )
                ax_ao += ax_ao.T
                r[idx] = C[:, si].T @ ax_ao @ C[:, sj]

            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.append(r.shape[-2])
                shape1.append(r.shape[-1])
                r.shape = shape1
            return r

        return fx

    def Ax0_Core(self, si, sj, sk, sl, cx=None, reshape=True):
        C = self.C
        nao = self.mol.nao
        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None
        if cx is None:
            cx = self.cx

        @timing
        @gccollect
        def fx(mo1_):
            mo1 = mo1_.copy()  # type: np.ndarray
            shape1 = list(mo1.shape)
            mo1.shape = (-1, shape1[-2], shape1[-1])
            if skl_none:
                dm1 = mo1
                if dm1.shape[-2] != self.nao or dm1.shape[-1] != self.nao:
                    raise ValueError("if `sk`, `sl` is None, we assume that mo1 passed in is an AO-based matrix!")
            else:
                dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.transpose((0, 2, 1))
            ax_ao = np.empty((dm1.shape[0], nao, nao))
            for idx, dmX in enumerate(dm1):
                ax_ao[idx] = (
                    + 1 * self.scf_eng.get_j(dm=dmX)
                    - 0.5 * cx * self.scf_eng.get_k(dm=dmX)
                )
            grdit = GridIterator(self.mol, self.grids, self.D, deriv=2, memory=self.grdit_memory)
            for grdh in grdit:
                kerh = KernelHelper(grdh, self.xc)
                for idx, dmX in enumerate(dm1):
                    tmp_K = np.einsum("kl, gl -> gk", dmX, grdh.ao_0)
                    rho_X_0 = np.einsum("gk, gk -> g", grdh.ao_0, tmp_K)
                    rho_X_1 = 2 * np.einsum("rgk, gk -> rg", grdh.ao_1, tmp_K)
                    gamma_XD = np.einsum("rg, rg -> g", rho_X_1, grdh.rho_1)
                    tmp_M = np.empty((4, grdh.ngrid))
                    tmp_M[0] = (
                        np.einsum("g, g -> g", rho_X_0, kerh.frr)
                        + 2 * np.einsum("g, g -> g", gamma_XD, kerh.frg)
                    )
                    tmp_M[1:4] = (
                        + 4 * np.einsum("g, g, rg -> rg", rho_X_0, kerh.frg, grdh.rho_1)
                        + 8 * np.einsum("g, g, rg -> rg", gamma_XD, kerh.fgg, grdh.rho_1)
                        + 4 * np.einsum("rg, g -> rg", rho_X_1, kerh.fg)
                    )
                    ax_ao[idx] += (
                        + np.einsum("rg, rgu, gv -> uv", tmp_M, grdh.ao[:4], grdh.ao_0)
                    )
                gc.collect()
            ax_ao += ax_ao.swapaxes(-1, -2)
            if sij_none:
                r = ax_ao
            else:
                r = C[:, si].T @ ax_ao @ C[:, sj]

            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.append(r.shape[-2])
                shape1.append(r.shape[-1])
                r.shape = shape1
            return r

        return fx

    @property
    def F_2_ao_GGAcontrib(self):
        if self._F_2_ao_GGAcontrib is None:
            self._F_2_ao_GGAcontrib = self._get_F_2_ao_GGAcontrib()
        return self._F_2_ao_GGAcontrib

    def _get_F_2_ao(self):
        return self.H_2_ao + self.F_2_ao_Jcontrib - 0.5 * self.cx * self.F_2_ao_Kcontrib + self.F_2_ao_GGAcontrib

    @timing
    @gccollect
    def _get_F_2_ao_GGAcontrib(self):
        natm = self.natm
        nao = self.nao

        F_2_ao_GGA = np.zeros((natm, natm, 3, 3, nao, nao))

        grdit = GridIterator(self.mol, self.grids, self.D, deriv=3, memory=self.grdit_memory)
        for grdh in grdit:
            kerh = KernelHelper(grdh, self.xc, deriv=3)
            pd_fr = kerh.frr * grdh.A_rho_1 + kerh.frg * grdh.A_gamma_1
            pd_fg = kerh.frg * grdh.A_rho_1 + kerh.fgg * grdh.A_gamma_1
            pd_rho_1 = grdh.A_rho_2
            pd_frr = kerh.frrr * grdh.A_rho_1 + kerh.frrg * grdh.A_gamma_1
            pd_frg = kerh.frrg * grdh.A_rho_1 + kerh.frgg * grdh.A_gamma_1
            pd_fgg = kerh.frgg * grdh.A_rho_1 + kerh.fggg * grdh.A_gamma_1
            pdpd_fr = (
                + np.einsum("Bsg, Atg -> ABtsg", pd_frr, grdh.A_rho_1)
                + np.einsum("Bsg, Atg -> ABtsg", pd_frg, grdh.A_gamma_1)
                + kerh.frr * grdh.AB_rho_2 + kerh.frg * grdh.AB_gamma_2
            )
            pdpd_fg = (
                + np.einsum("Bsg, Atg -> ABtsg", pd_frg, grdh.A_rho_1)
                + np.einsum("Bsg, Atg -> ABtsg", pd_fgg, grdh.A_gamma_1)
                + kerh.frg * grdh.AB_rho_2 + kerh.fgg * grdh.AB_gamma_2
            )
            pdpd_rho_1 = grdh.AB_rho_3

            # Contrib 1
            contrib1 = (
                + 0.5 * np.einsum("ABtsg, gu, gv -> ABtsuv", pdpd_fr, grdh.ao_0, grdh.ao_0)
                + 2 * np.einsum("ABtsg, rg, rgu, gv -> ABtsuv", pdpd_fg, grdh.rho_1, grdh.ao_1, grdh.ao_0)
                + 2 * np.einsum("Atg, Bsrg, rgu, gv -> ABtsuv", pd_fg, pd_rho_1, grdh.ao_1, grdh.ao_0)
                + 2 * np.einsum("Bsg, Atrg, rgu, gv -> ABtsuv", pd_fg, pd_rho_1, grdh.ao_1, grdh.ao_0)
                + 2 * np.einsum("g, ABtsrg, rgu, gv -> ABtsuv", kerh.fg, pdpd_rho_1, grdh.ao_1, grdh.ao_0)
            )
            contrib1 += contrib1.swapaxes(-1, -2)
            F_2_ao_GGA += contrib1

            # Contrib 2
            tmp_contrib = (
                - np.einsum("Bsg, tgu, gv -> Btsuv", pd_fr, grdh.ao_1, grdh.ao_0)
                - 2 * np.einsum("Bsg, rg, tgu, rgv -> Btsuv", pd_fg, grdh.rho_1, grdh.ao_1, grdh.ao_1)
                - 2 * np.einsum("Bsg, rg, trgu, gv -> Btsuv", pd_fg, grdh.rho_1, grdh.ao_2, grdh.ao_0)
                - 2 * np.einsum("g, Bsrg, tgu, rgv -> Btsuv", kerh.fg, pd_rho_1, grdh.ao_1, grdh.ao_1)
                - 2 * np.einsum("g, Bsrg, trgu, gv -> Btsuv", kerh.fg, pd_rho_1, grdh.ao_2, grdh.ao_0)
            )
            contrib2 = np.zeros((natm, natm, 3, 3, nao, nao))
            for A in range(natm):
                sA = self.mol_slice(A)
                contrib2[A, :, :, :, sA] += tmp_contrib[:, :, :, sA]
            contrib2 += contrib2.transpose((0, 1, 2, 3, 5, 4))
            contrib2 += contrib2.transpose((1, 0, 3, 2, 4, 5))
            F_2_ao_GGA += contrib2

            # Contrib 3
            contrib3 = np.zeros((natm, natm, 3, 3, nao, nao))

            tmp_contrib = (
                + np.einsum("g, tsgu, gv -> tsuv", kerh.fr, grdh.ao_2, grdh.ao_0)
                + 2 * np.einsum("g, rg, tsrgu, gv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_3, grdh.ao_0)
                + 2 * np.einsum("g, rg, tsgu, rgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_2, grdh.ao_1)
            )
            for A in range(natm):
                sA = self.mol_slice(A)
                contrib3[A, A, :, :, sA] += tmp_contrib[:, :, sA]
            tmp_contrib = (
                + np.einsum("g, tgu, sgv -> tsuv", kerh.fr, grdh.ao_1, grdh.ao_1)
                + 2 * np.einsum("g, rg, trgu, sgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_2, grdh.ao_1)
                + 2 * np.einsum("g, rg, tgu, srgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_1, grdh.ao_2)
            )
            for A in range(natm):
                for B in range(natm):
                    sA, sB = self.mol_slice(A), self.mol_slice(B)
                    contrib3[A, B, :, :, sA, sB] += tmp_contrib[:, :, sA, sB]
            contrib3 += contrib3.swapaxes(-1, -2)
            F_2_ao_GGA += contrib3

        # Finalize
        return F_2_ao_GGA

    def Ax1_Core(self, si, sj, sk, sl, cx=None):
        if cx is None:
            cx = self.cx
        C, Co = self.C, self.Co
        natm, nao, nmo, nocc = self.natm, self.nao, self.nmo, self.nocc
        so, sv = self.so, self.sv
        U_pi_fake = np.empty((natm, 3, nmo, nocc))
        U_pi_fake[:, :, so, so] = - 0.5 * self.S_1_mo[:, :, so, so]
        U_pi_fake[:, :, sv, so] = self.U_1_vo
        dmU = C @ U_pi_fake @ Co.T
        dmU += dmU.swapaxes(-1, -2)

        @timing
        def fx(mo1):
            dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.swapaxes(-1, -2)
            ax_final = np.zeros((natm, dm1.shape[0], 3, dm1.shape[1], si.stop - si.start, sj.stop - sj.start))
            for B in range(dm1.shape[0]):
                for s in range(dm1.shape[1]):
                    dmX = dm1[B, s]

                    grdit = GridIterator(self.mol, self.grids, self.D, deriv=3, memory=self.grdit_memory)
                    for grdh in grdit:
                        kerh = KernelHelper(grdh, self.xc, deriv=3)
                        # Define some kernel and density derivative alias
                        pd_frr = kerh.frrr * grdh.A_rho_1 + kerh.frrg * grdh.A_gamma_1
                        pd_frg = kerh.frrg * grdh.A_rho_1 + kerh.frgg * grdh.A_gamma_1
                        pd_fgg = kerh.frgg * grdh.A_rho_1 + kerh.fggg * grdh.A_gamma_1
                        pd_fg = kerh.frg * grdh.A_rho_1 + kerh.fgg * grdh.A_gamma_1
                        pd_rho_1 = grdh.A_rho_2

                        # Form dmX density grid
                        rho_X_0 = grdh.get_rho_0(dmX)
                        rho_X_1 = grdh.get_rho_1(dmX)
                        pd_rho_X_0 = grdh.get_A_rho_1(dmX)
                        pd_rho_X_1 = grdh.get_A_rho_2(dmX)

                        # Define temporary intermediates
                        tmp_M_0 = (
                            + np.einsum("g, g -> g", kerh.frr, rho_X_0)
                            + 2 * np.einsum("g, wg, wg -> g", kerh.frg, grdh.rho_1, rho_X_1)
                        )
                        tmp_M_1 = (
                            + 4 * np.einsum("g, g, rg -> rg", kerh.frg, rho_X_0, grdh.rho_1)
                            + 8 * np.einsum("g, wg, wg, rg -> rg", kerh.fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                            + 4 * np.einsum("g, rg -> rg", kerh.fg, rho_X_1)
                        )
                        pd_tmp_M_0 = (
                            + np.einsum("Atg, g -> Atg", pd_frr, rho_X_0)
                            + np.einsum("g, Atg -> Atg", kerh.frr, pd_rho_X_0)
                            + 2 * np.einsum("Atg, wg, wg -> Atg", pd_frg, grdh.rho_1, rho_X_1)
                            + 2 * np.einsum("g, Atwg, wg -> Atg", kerh.frg, pd_rho_1, rho_X_1)
                            + 2 * np.einsum("g, wg, Atwg -> Atg", kerh.frg, grdh.rho_1, pd_rho_X_1)
                        )
                        pd_tmp_M_1 = (
                            + 4 * np.einsum("Atg, g, rg -> Atrg", pd_frg, rho_X_0, grdh.rho_1)
                            + 4 * np.einsum("g, g, Atrg -> Atrg", kerh.frg, rho_X_0, pd_rho_1)
                            + 4 * np.einsum("g, Atg, rg -> Atrg", kerh.frg, pd_rho_X_0, grdh.rho_1)
                            + 8 * np.einsum("Atg, wg, wg, rg -> Atrg", pd_fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                            + 8 * np.einsum("g, Atwg, wg, rg -> Atrg", kerh.fgg, pd_rho_1, rho_X_1, grdh.rho_1)
                            + 8 * np.einsum("g, wg, wg, Atrg -> Atrg", kerh.fgg, grdh.rho_1, rho_X_1, pd_rho_1)
                            + 8 * np.einsum("g, wg, Atwg, rg -> Atrg", kerh.fgg, grdh.rho_1, pd_rho_X_1, grdh.rho_1)
                            + 4 * np.einsum("Atg, rg -> Atrg", pd_fg, rho_X_1)
                            + 4 * np.einsum("g, Atrg -> Atrg", kerh.fg, pd_rho_X_1)
                        )

                        contrib1 = np.zeros((natm, 3, nao, nao))
                        contrib1 += np.einsum("Atg, gu, gv -> Atuv", pd_tmp_M_0, grdh.ao_0, grdh.ao_0)
                        contrib1 += np.einsum("Atrg, rgu, gv -> Atuv", pd_tmp_M_1, grdh.ao_1, grdh.ao_0)
                        contrib1 += contrib1.swapaxes(-1, -2)

                        tmp_contrib = (
                            - 2 * np.einsum("g, tgu, gv -> tuv", tmp_M_0, grdh.ao_1, grdh.ao_0)
                            - np.einsum("rg, trgu, gv -> tuv", tmp_M_1, grdh.ao_2, grdh.ao_0)
                            - np.einsum("rg, tgu, rgv -> tuv", tmp_M_1, grdh.ao_1, grdh.ao_1)
                        )

                        contrib2 = np.zeros((natm, 3, nao, nao))
                        for A in range(natm):
                            sA = self.mol_slice(A)
                            contrib2[A, :, sA] += tmp_contrib[:, sA]

                        contrib2 += contrib2.swapaxes(-1, -2)

                        # U contribution to \partial_{A_t} A
                        rho_U_0 = np.einsum("Atuv, gu, gv -> Atg", dmU, grdh.ao_0, grdh.ao_0)
                        rho_U_1 = 2 * np.einsum("Atuv, rgu, gv -> Atrg", dmU, grdh.ao_1, grdh.ao_0)
                        gamma_U_0 = 2 * np.einsum("rg, Atrg -> Atg", grdh.rho_1, rho_U_1)
                        pdU_frr = kerh.frrr * rho_U_0 + kerh.frrg * gamma_U_0
                        pdU_frg = kerh.frrg * rho_U_0 + kerh.frgg * gamma_U_0
                        pdU_fgg = kerh.frgg * rho_U_0 + kerh.fggg * gamma_U_0
                        pdU_fg = kerh.frg * rho_U_0 + kerh.fgg * gamma_U_0
                        pdU_rho_1 = rho_U_1
                        pdU_tmp_M_0 = (
                            + np.einsum("Atg, g -> Atg", pdU_frr, rho_X_0)
                            + 2 * np.einsum("Atg, wg, wg -> Atg", pdU_frg, grdh.rho_1, rho_X_1)
                            + 2 * np.einsum("g, Atwg, wg -> Atg", kerh.frg, pdU_rho_1, rho_X_1)
                        )
                        pdU_tmp_M_1 = (
                            + 4 * np.einsum("Atg, g, rg -> Atrg", pdU_frg, rho_X_0, grdh.rho_1)
                            + 4 * np.einsum("g, g, Atrg -> Atrg", kerh.frg, rho_X_0, pdU_rho_1)
                            + 8 * np.einsum("Atg, wg, wg, rg -> Atrg", pdU_fgg, grdh.rho_1, rho_X_1, grdh.rho_1)
                            + 8 * np.einsum("g, Atwg, wg, rg -> Atrg", kerh.fgg, pdU_rho_1, rho_X_1, grdh.rho_1)
                            + 8 * np.einsum("g, wg, wg, Atrg -> Atrg", kerh.fgg, grdh.rho_1, rho_X_1, pdU_rho_1)
                            + 4 * np.einsum("Atg, rg -> Atrg", pdU_fg, rho_X_1)
                        )

                        contrib3 = np.zeros((natm, 3, nao, nao))
                        contrib3 += np.einsum("Atg, gu, gv -> Atuv", pdU_tmp_M_0, grdh.ao_0, grdh.ao_0)
                        contrib3 += np.einsum("Atrg, rgu, gv -> Atuv", pdU_tmp_M_1, grdh.ao_1, grdh.ao_0)
                        contrib3 += contrib3.swapaxes(-1, -2)

                        # Finalize
                        ax_final[:, B, :, s] += C[:, si].T @ (contrib1 + contrib2 + contrib3) @ C[:, sj]
            ax_final += super(GGAHelper, self).Ax1_Core(si, sj, sk, sl, cx=cx)(mo1)
            return ax_final

        return fx


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

    nmo = mol.nao
    nocc = mol.nelec[0]
    so = slice(0, nocc)
    sa = slice(0, nmo)

    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.becke_scheme = dft.gen_grid.stratmann
    grids.build()

    ggah = GGAHelper(mol, "b3lypg", grids)

    X = np.random.random((3, nmo, nocc))
    print(np.allclose(
        hessian.rhf.gen_vind(ggah.scf_eng, ggah.C, ggah.mo_occ)(X),
        ggah.Ax0_Core(sa, so, sa, so)(X)
    ))


if __name__ == "__main__":
    main()
