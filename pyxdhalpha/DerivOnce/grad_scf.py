import numpy as np
from functools import partial
import os

from pyscf import grad
from pyscf.scf import _vhf

from pyxdhalpha.DerivOnce.deriv_once_scf import DerivOnceSCF, DerivOnceNCDFT
from pyxdhalpha.Utilities import GridIterator, KernelHelper, timing

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


# Cubic Inheritance: A2
class GradSCF(DerivOnceSCF):

    def Ax1_Core(self, si, sj, sk, sl, reshape=True):

        C, Co = self.C, self.Co
        natm, nao, nmo, nocc = self.natm, self.nao, self.nmo, self.nocc
        mol = self.mol
        cx = self.cx
        so, sv = self.so, self.sv

        dmU = C @ self.U_1[:, :, so] @ Co.T
        dmU += dmU.swapaxes(-1, -2)
        dmU.shape = (natm, 3, nao, nao)

        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None

        @timing
        def fx(X_):
            if X_ is 0:
                return 0
            X = X_.copy()  # type: np.ndarray
            shape1 = list(X.shape)
            X.shape = (-1, shape1[-2], shape1[-1])
            if skl_none:
                dm = X
                if dm.shape[-2] != nao or dm.shape[-1] != nao:
                    raise ValueError("if `sk`, `sl` is None, we assume that mo1 passed in is an AO-based matrix!")
            else:
                dm = C[:, sk] @ X @ C[:, sl].T
            dm += dm.transpose((0, 2, 1))

            ax_ao = np.empty((natm, 3, dm.shape[0], nao, nao))

            # Actual calculation
            for B in range(dm.shape[0]):
                dmX = dm[B]
                # (ut v | k l), (ut k | v l)
                j_1, k_1 = _vhf.direct_mapdm(
                    mol._add_suffix('int2e_ip1'), "s2kl",
                    ("lk->s1ij", "jk->s1il"),
                    dmX, 3,
                    mol._atm, mol._bas, mol._env
                )

                # HF Part
                for A in range(natm):
                    ax = np.zeros((3, nao, nao))
                    shl0, shl1, p0, p1 = mol.aoslice_by_atom()[A]
                    sA = slice(p0, p1)  # equivalent to mol_slice(A)
                    ax[:, sA, :] -= 2 * j_1[:, sA, :]
                    ax[:, :, sA] -= 2 * j_1[:, sA, :].swapaxes(-1, -2)
                    ax[:, sA, :] += cx * k_1[:, sA, :]
                    ax[:, :, sA] += cx * k_1[:, sA, :].swapaxes(-1, -2)
                    # (kt l | u v), (kt u | l v)
                    j_1A, k_1A = _vhf.direct_mapdm(
                        mol._add_suffix('int2e_ip1'), "s2kl",
                        ("ji->s1kl", "li->s1kj"),
                        dmX[:, p0:p1], 3,
                        mol._atm, mol._bas, mol._env,
                        shls_slice=((shl0, shl1) + (0, mol.nbas) * 3)
                    )
                    ax -= 4 * j_1A
                    ax += cx * (k_1A + k_1A.swapaxes(-1, -2))

                    ax_ao[A, :, B] = ax

                # GGA Part
                if self.xc_type == "GGA":
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

                        ax_ao[:, :, B] += contrib1 + contrib2 + contrib3

            ax_ao.shape = (natm * 3, dm.shape[0], nao, nao)

            if not sij_none:
                ax_ao = np.einsum("ABuv, ui, vj -> ABij", ax_ao, C[:, si], C[:, sj])
            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.insert(0, ax_ao.shape[0])
                shape1.append(ax_ao.shape[-2])
                shape1.append(ax_ao.shape[-1])
                ax_ao.shape = shape1

            return ax_ao

        return fx

    def _get_H_1_ao(self):
        return np.array([self.scf_grad.hcore_generator()(A) for A in range(self.natm)])\
            .reshape((-1, self.nao, self.nao))

    def _get_F_1_ao(self):
        return np.array(self.scf_hess.make_h1(self.C, self.mo_occ)).reshape((-1, self.nao, self.nao))

    def _get_S_1_ao(self):
        int1e_ipovlp = self.mol.intor("int1e_ipovlp")

        def get_S_S_ao(A):
            ao_matrix = np.zeros((3, self.nao, self.nao))
            sA = self.mol_slice(A)
            ao_matrix[:, sA] = -int1e_ipovlp[:, sA]
            return ao_matrix + ao_matrix.swapaxes(1, 2)

        S_1_ao = np.array([get_S_S_ao(A) for A in range(self.natm)]).reshape((-1, self.nao, self.nao))
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
        return eri1_ao.reshape((-1, self.nao, self.nao, self.nao, self.nao))
    
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


# Cubic Inheritance: B2
class GradNCDFT(DerivOnceNCDFT, GradSCF):

    @property
    def DerivOnceMethod(self):
        return GradSCF

    def _get_E_1(self):
        natm = self.natm
        so, sv, sa = self.so, self.sv, self.sa
        B_1 = self.B_1
        Z = self.Z
        E_1 = 4 * np.einsum("ai, Aai -> A", Z, B_1[:, sv, so]).reshape((natm, 3))
        E_1 += self.nc_deriv.E_1
        return E_1


class Test_GradSCF:

    def test_HF_grad(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2()
        config = {
            "scf_eng": H2O2.hf_eng
        }
        helper = GradSCF(config)
        hf_grad = helper.scf_grad

        assert(np.allclose(
            helper.E_1, hf_grad.grad(),
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-HF-freq.fchk"))

        assert(np.allclose(
            helper.E_1, formchk.grad(),
            atol=1e-6, rtol=1e-4
        ))

    def test_B3LYP_grad(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2()
        config = {
            "scf_eng": H2O2.gga_eng
        }
        helper = GradSCF(config)
        gga_grad = helper.scf_grad

        assert(np.allclose(
            helper.E_1, gga_grad.grad(),
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-B3LYP-freq.fchk"))

        # TODO: This is a weaker compare! Try to modulize that someday.
        assert(np.allclose(
            helper.E_1, formchk.grad(),
            atol=1e-5, rtol=1e-4
        ))

    def test_HF_B3LYP_grad(self):

        import pickle
        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2

        H2O2 = Mol_H2O2()
        config = {
            "scf_eng": H2O2.hf_eng,
            "nc_eng": H2O2.gga_eng
        }
        helper = GradNCDFT(config)

        with open(resource_filename("pyxdhalpha", "Validation/numerical_deriv/ncdft_derivonce_hf_b3lyp.dat"),
                  "rb") as f:
            ref_grad = pickle.load(f)["grad"].reshape(-1, 3)

        assert (np.allclose(
            helper.E_1, ref_grad,
            atol=1e-6, rtol=1e-4
        ))
