from hf_helper import HFHelper
from gga_helper import GGAHelper
from grid_iterator import GridIterator
from grid_helper import KernelHelper
import numpy as np
from functools import partial
from pyscf import scf
import pyscf.scf.cphf
import os

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class NCGGAEngine:

    def __init__(self, scfh, nch):

        self.scfh = scfh  # type: GGAHelper
        self.nch = nch  # type: GGAHelper

        self.nch.C = self.scfh.C
        self.nch.mo_occ = self.scfh.mo_occ
        self.nch.D = self.scfh.D

        self.E_0 = None
        self.E_1 = None
        self.E_2 = None
        self.E_SS = None
        self.E_SU = None
        self.E_UU = None

        return

    def get_E_0(self):

        self.E_0 = self.nch.scf_eng.energy_tot(dm=self.scfh.D)
        return self.E_0

    def get_E_1(self):

        scfh = self.scfh
        nch = self.nch
        D = scfh.D
        sv = scfh.sv
        so = scfh.so
        natm = scfh.natm

        E_S = np.einsum("Atuv, uv -> At", scfh.H_1_ao, D)
        grdit = GridIterator(scfh.mol, nch.grids, D, deriv=2)
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

        Z_1 = scf.cphf.solve(scfh.Ax0_Core(sv, so, sv, so), scfh.e, scfh.mo_occ, nch.F_0_mo[sv, so])[0]
        E_U = (
            + 4 * np.einsum("Atai, ai -> At", scfh.B_1[:, :, sv, so], Z_1)
            - 2 * np.einsum("Atki, ki -> At", scfh.S_1_mo[:, :, so, so], nch.F_0_mo[so, so])
        )

        E_1 = E_S + E_U + nch.scf_grad.grad_nuc()
        self.E_1 = E_1
        return self.E_1

    def get_E_2(self):

        if self.E_SS is None:
            self.get_E_SS()
        if self.E_SU is None:
            self.get_E_SU()
        if self.E_UU is None:
            self.get_E_UU()

        self.E_2 = self.E_SS + self.E_SU + self.E_UU + self.nch.scf_hess.hess_nuc()

        return self.E_2

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
        grdit = GridIterator(scfh.mol, nch.grids, D, deriv=3)
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
            + 0.5 * np.einsum("ABtsuv, uv -> ABts", scfh.get_F_2_ao_JKcontrib_(cx=nch.cx), D)
        )

        E_SS = E_SS_GGA_contrib1 + E_SS_GGA_contrib2 + E_SS_GGA_contrib3 + E_SS_HF_contrib
        self.E_SS = E_SS
        return self.E_SS

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
        self.E_SU = E_SU
        return self.E_SU

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

        Z = scf.cphf.solve(Ax0_Core(sv, so, sv, so), e, scfh.mo_occ, nch.F_0_mo[sv, so])[0]

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
        E_UU_safe_4 = 4 * np.einsum("ai, ABtsai -> ABts", Z, B_2_new_vo)
        E_UU_safe_4 += E_UU_safe_4.transpose((1, 0, 3, 2))

        E_UU = E_UU_safe_1 + E_UU_safe_2 + E_UU_safe_3 + E_UU_safe_4
        self.E_UU = E_UU

        return self.E_UU


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
