import numpy as np
from functools import partial
from pyscf import scf, lib
import pyscf.scf.cphf
import os, warnings

from hessian import HFHelper, GGAHelper
from utilities import GridIterator, KernelHelper, timing, gccollect


MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class NCGGAEngine:

    def __init__(self, scfh, nch, grdit_memory=2000):

        self.scfh = scfh  # type: GGAHelper
        self.nch = nch  # type: GGAHelper
        self.grdit_memory = grdit_memory

        self.nch.C = self.scfh.C
        self.nch.mo_occ = self.scfh.mo_occ
        self.nch.D = self.scfh.D

        self._E_0 = None
        self._E_1 = None
        self._E_2 = None
        self._E_SS = None
        self._E_SU = None
        self._E_UU = None
        self._Z_1 = None

        return

    @property
    def E_0(self):
        if self._E_0 is None:
            self._E_0 = self.get_E_0()
        return self._E_0

    @property
    def E_1(self):
        if self._E_1 is None:
            self._E_1 = self.get_E_1()
        return self._E_1

    @property
    def E_SS(self):
        if self._E_SS is None:
            self._E_SS = self.get_E_SS()
        return self._E_SS

    @property
    def E_SU(self):
        if self._E_SU is None:
            self._E_SU = self.get_E_SU()
        return self._E_SU

    @property
    def E_UU(self):
        if self._E_UU is None:
            self._E_UU = self.get_E_UU()
        return self._E_UU

    @property
    def E_2(self):
        if self._E_2 is None:
            self._E_2 = self.get_E_2()
        return self._E_2

    @property
    def Z_1(self):
        if self._Z_1 is None:
            self._Z_1 = self._get_Z_1()
        return self._Z_1

    @timing
    @gccollect
    def get_E_0(self):

        E_0 = self.nch.scf_eng.energy_tot(dm=self.scfh.D)
        return E_0

    @timing
    @gccollect
    def _get_Z_1(self):
        scfh, nch = self.scfh, self.nch
        so, sv = scfh.so, scfh.sv
        Z_1 = scf.cphf.solve(scfh.Ax0_Core(sv, so, sv, so), scfh.e, scfh.mo_occ, nch.F_0_mo[sv, so],
                             max_cycle=100, tol=1e-13)[0]

        # Test whether converged
        conv = (
            + Z_1 * lib.direct_sum("a - i", scfh.ev, scfh.eo)
            + scfh.Ax0_Core(sv, so, sv, so)(Z_1)
            + nch.F_0_mo[sv, so]
        )
        if abs(conv).max() > 1e-8:
            msg = "\nget_E_1: CP-HF not converged!\nMaximum deviation: " + str(abs(conv).max())
            warnings.warn(msg)

        return Z_1

    @timing
    @gccollect
    def get_E_1(self):

        scfh = self.scfh
        nch = self.nch
        D = scfh.D
        sv = scfh.sv
        so = scfh.so
        natm = scfh.natm

        E_S = np.einsum("Atuv, uv -> At", scfh.H_1_ao, D)
        grdit = GridIterator(scfh.mol, nch.grids, D, deriv=2, memory=self.grdit_memory)
        for grdh in grdit:
            kerh = KernelHelper(grdh, nch.xc)
            E_S += (
                + np.einsum("g, Atg -> At", kerh.fr, grdh.A_rho_1)
                + 2 * np.einsum("g, rg, Atrg -> At", kerh.fg, grdh.rho_1, grdh.A_rho_2)
            )

        # From memory consumption point, we use higher subroutines in PySCF to generate ERI contribution
        jk_1 = (
            + 2 * scfh.scf_grad.get_j(dm=D)
            - nch.cx * scfh.scf_grad.get_k(dm=D)
        )
        for A in range(natm):
            sA = scfh.mol_slice(A)
            E_S[A] += np.einsum("tuv, uv -> t", jk_1[:, sA], D[sA])

        E_U = (
            + 4 * np.einsum("Atai, ai -> At", scfh.B_1[:, :, sv, so], self.Z_1)
            - 2 * np.einsum("Atki, ki -> At", scfh.S_1_mo[:, :, so, so], nch.F_0_mo[so, so])
        )

        E_1 = E_S + E_U + nch.scf_grad.grad_nuc()
        return E_1

    @timing
    def get_E_2(self):
        E_2 = self.E_SS + self.E_SU + self.E_UU + self.nch.scf_hess.hess_nuc()
        return E_2

    @timing
    @gccollect
    def get_E_SS(self):

        scfh = self.scfh
        nch = self.nch
        mol_slice = scfh.mol_slice
        natm = scfh.natm
        D = scfh.D

        # GGA Contribution
        E_SS_GGA_contrib1 = np.zeros((natm, natm, 3, 3))
        E_SS_GGA_contrib2 = np.zeros((natm, natm, 3, 3))
        E_SS_GGA_contrib3 = np.zeros((natm, natm, 3, 3))
        grdit = GridIterator(scfh.mol, nch.grids, D, deriv=3, memory=self.grdit_memory)
        for grdh in grdit:
            kerh = KernelHelper(grdh, nch.xc)

            tmp_tensor_1 = (
                + 2 * np.einsum("g, Tgu, gv -> Tuv", kerh.fr, grdh.ao_2T, grdh.ao_0)
                + 4 * np.einsum("g, rg, rTgu, gv -> Tuv", kerh.fg, grdh.rho_1, grdh.ao_3T, grdh.ao_0)
                + 4 * np.einsum("g, rg, Tgu, rgv -> Tuv", kerh.fg, grdh.rho_1, grdh.ao_2T, grdh.ao_1)
            )
            XX, XY, XZ, YY, YZ, ZZ = range(6)
            for A in range(natm):
                sA = mol_slice(A)
                E_SS_GGA_contrib1[A, A] += np.einsum("Tuv, uv -> T", tmp_tensor_1[:, sA], D[sA])[
                    [XX, XY, XZ, XY, YY, YZ, XZ, YZ, ZZ]].reshape(3, 3)

            tmp_tensor_2 = 4 * np.einsum("g, rg, trgu, sgv -> tsuv", kerh.fg, grdh.rho_1, grdh.ao_2, grdh.ao_1)
            tmp_tensor_2 += tmp_tensor_2.transpose((1, 0, 3, 2))
            tmp_tensor_2 += 2 * np.einsum("g, tgu, sgv -> tsuv", kerh.fr, grdh.ao_1, grdh.ao_1)
            E_SS_GGA_contrib2_inbatch = np.zeros((natm, natm, 3, 3))
            for A in range(natm):
                sA = mol_slice(A)
                for B in range(A + 1):
                    sB = mol_slice(B)
                    E_SS_GGA_contrib2_inbatch[A, B] += np.einsum("tsuv, uv -> ts",
                                                                 tmp_tensor_2[:, :, sA, sB], D[sA, sB])
                    if A != B:
                        E_SS_GGA_contrib2_inbatch[B, A] += E_SS_GGA_contrib2_inbatch[A, B].T
            E_SS_GGA_contrib2 += E_SS_GGA_contrib2_inbatch

            E_SS_GGA_contrib3 += (
                + np.einsum("g, Atg, Bsg -> ABts", kerh.frr, grdh.A_rho_1, grdh.A_rho_1)
                + 2 * np.einsum("g, wg, Atwg, Bsg -> ABts", kerh.frg, grdh.rho_1, grdh.A_rho_2, grdh.A_rho_1)
                + 2 * np.einsum("g, Atg, rg, Bsrg -> ABts", kerh.frg, grdh.A_rho_1, grdh.rho_1, grdh.A_rho_2)
                + 4 * np.einsum("g, wg, Atwg, rg, Bsrg -> ABts", kerh.fgg, grdh.rho_1, grdh.A_rho_2, grdh.rho_1,
                                grdh.A_rho_2)
                + 2 * np.einsum("g, Atrg, Bsrg -> ABts", kerh.fg, grdh.A_rho_2, grdh.A_rho_2)
            )

        # HF Contribution
        E_SS_HF_contrib = (
            + np.einsum("ABtsuv, uv -> ABts", scfh.H_2_ao, D)
            + 0.5 * np.einsum("ABtsuv, uv -> ABts", scfh.F_2_ao_Jcontrib - 0.5 * nch.cx * scfh.F_2_ao_Kcontrib, D)
        )

        E_SS = E_SS_GGA_contrib1 + E_SS_GGA_contrib2 + E_SS_GGA_contrib3 + E_SS_HF_contrib
        return E_SS

    @timing
    @gccollect
    def get_E_SU(self):

        nch = self.nch

        U_1_vo = self.scfh.U_1_vo
        S_1_mo = self.scfh.S_1_mo
        sv = self.scfh.sv
        so = self.scfh.so

        E_SU = (
            + 4 * np.einsum("Atpi, Bspi -> ABts", U_1_vo, nch.F_1_mo[:, :, sv, so])
            - 2 * np.einsum("Atpi, Bspi -> ABts", S_1_mo[:, :, so, so], nch.F_1_mo[:, :, so, so])
        )
        E_SU += E_SU.transpose((1, 0, 3, 2))
        return E_SU

    @timing
    @gccollect
    def get_E_UU(self):

        scfh = self.scfh
        nch = self.nch

        natm = scfh.natm
        nmo = scfh.nmo
        nocc = scfh.nocc
        sa, so, sv = scfh.sa, scfh.so, scfh.sv

        Ax0_Core = scfh.Ax0_Core
        Ax1_Core = scfh.Ax1_Core
        F_1_mo = scfh.F_1_mo
        S_1_mo = scfh.S_1_mo
        F_2_ao = scfh.F_2_ao
        S_2_mo = scfh.S_2_mo
        e, eo, ev = scfh.e, scfh.eo, scfh.ev
        U_1_vo = scfh.U_1_vo
        U_1_ov = scfh.U_1_ov
        C, Co, Cv = scfh.C, scfh.Co, scfh.Cv

        U_pi_fake = np.empty((natm, 3, nmo, nocc))
        U_pi_fake[:, :, so, so] = - 0.5 * S_1_mo[:, :, so, so]
        U_pi_fake[:, :, sv, so] = U_1_vo

        E_UU_safe_1 = 4 * np.einsum("Atpi, Bspi -> ABts", U_pi_fake, nch.Ax0_Core(sa, so, sa, so)(U_pi_fake))

        E_UU_safe_2 = 4 * np.einsum("Atai, Bsbi, ab -> ABts", U_1_vo, U_1_vo, nch.F_0_mo[sv, sv])

        E_UU_safe_3 = (
            - 1 * np.einsum("ABtsij, ij -> ABts", S_2_mo[:, :, :, :, so, so], nch.F_0_mo[so, so])
            - 2 * np.einsum("Atia, Bsja, ij -> ABts", U_1_ov, U_1_ov, nch.F_0_mo[so, so])
            + 2 * np.einsum("Atim, Bsjm, ij -> ABts", S_1_mo[:, :, so, :], S_1_mo[:, :, so, :], nch.F_0_mo[so, so])
        )
        E_UU_safe_3 += E_UU_safe_3.transpose((1, 0, 3, 2))

        B_2_tmp_1 = (
            Co @ (
                - 0.25 * S_2_mo[:, :, :, :, so, so]
                + 0.5 * np.einsum("Atkm, Bslm -> ABtskl", S_1_mo[:, :, so, :], S_1_mo[:, :, so, :])
                - 0.5 * np.einsum("Atkb, Bslb -> ABtskl", U_1_ov, U_1_ov)
            ) @ Co.T
            + 0.5 * Cv @ np.einsum("Atbk, Bsck -> ABtsbc", U_1_vo, U_1_vo) @ Cv.T
        )
        B_2_tmp_2 = Ax0_Core(None, None, sa, so)(U_pi_fake)

        B_2_new_vo = (
            + 0.5 * np.einsum("ABtsuv, ua, vi -> ABtsai", F_2_ao, Cv, Co)
            - 0.5 * np.einsum("ABtsai, i -> ABtsai", S_2_mo[:, :, :, :, sv, so], eo)
            + np.einsum("Atim, Bsam, i -> ABtsai", S_1_mo[:, :, so, :], S_1_mo[:, :, sv, :], eo)
            + np.einsum("Atka, Bski -> ABtsai", U_1_ov, F_1_mo[:, :, so, so])
            + np.einsum("Atbi, Bsba -> ABtsai", U_1_vo, F_1_mo[:, :, sv, sv])
            + Ax0_Core(sv, so, None, None)(B_2_tmp_1)
            + np.einsum("Atka, uk, vi, Bsuv -> ABtsai", U_1_ov, Co, Co, B_2_tmp_2)
            + np.einsum("Atbi, ub, va, Bsuv -> ABtsai", U_1_vo, Cv, Cv, B_2_tmp_2)
            + Ax1_Core(sv, so, sa, so)(U_pi_fake)
            + np.einsum("Bsib, Atba, i -> ABtsai", U_1_ov, S_1_mo[:, :, sv, sv], eo)
            + np.einsum("Atak, Bski, a -> ABtsai", U_1_vo, S_1_mo[:, :, so, so], ev)
        )

        # B_2_new_vo += B_2_new_vo.transpose((1, 0, 3, 2, 4, 5))
        E_UU_safe_4 = 4 * np.einsum("ai, ABtsai -> ABts", self.Z_1, B_2_new_vo)
        E_UU_safe_4 += E_UU_safe_4.transpose((1, 0, 3, 2))

        E_UU = E_UU_safe_1 + E_UU_safe_2 + E_UU_safe_3 + E_UU_safe_4

        return E_UU


if __name__ == "__main__":

    from pyscf import gto, dft

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

    scfh_ = HFHelper(mol)
    nch_ = GGAHelper(mol, "b3lypg", grids, init_scf=False)
    ncengine = NCGGAEngine(scfh_, nch_)

    print(ncengine.get_E_0())
    print(ncengine.get_E_1())
    print(ncengine.get_E_2())
