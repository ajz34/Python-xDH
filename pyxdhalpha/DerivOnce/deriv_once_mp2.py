import numpy as np
from abc import ABC
from functools import partial
import os
import warnings

from pyscf.scf import cphf

from pyxdhalpha.DerivOnce import DerivOnce

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DerivOnceMP2(DerivOnce, ABC):

    def __init__(self, scf_eng, rotation=True, grdit_memory=2000):
        super(DerivOnceMP2, self).__init__(scf_eng, rotation, grdit_memory)
        self._t_iajb = NotImplemented
        self._T_iajb = NotImplemented
        self._L = NotImplemented
        self._D_r = NotImplemented
        self._D_r_ai_flag = False  # Flag indicates that D_r's ai part has been generated
        self._W_I = NotImplemented
        self._D_iajb = NotImplemented

    # region Properties

    @property
    def t_iajb(self):
        if self._t_iajb is NotImplemented:
            self._t_iajb = self._get_t_iajb()
        return self._t_iajb

    @property
    def T_iajb(self):
        if self._T_iajb is NotImplemented:
            self._T_iajb = self._get_T_iajb()
        return self._T_iajb

    @property
    def L(self):
        if self._L is NotImplemented:
            self._L = self._get_L()
        return self._L

    @property
    def D_r(self):
        if self._D_r is NotImplemented or not self._D_r_ai_flag:
            self._D_r = self._get_D_r()
            self._D_r_ai_flag = True
        return self._D_r

    @property
    def W_I(self):
        if self._W_I is NotImplemented:
            self._W_I = self._get_W_I()
        return self._W_I

    @property
    def D_iajb(self):
        if self._D_iajb is NotImplemented:
            self._D_iajb = self._get_D_iajb()
        return self._D_iajb

    # endregion

    # region Setter

    def _get_t_iajb(self):
        so, sv = self.so, self.sv
        return self.eri0_mo[so, sv, so, sv] / self.D_iajb

    def _get_T_iajb(self):
        return 2 * self.t_iajb - self.t_iajb.swapaxes(-1, -3)

    def _get_L(self):
        nvir, nocc, nmo = self.nvir, self.nocc, self.nmo
        so, sv, sa = self.so, self.sv, self.sa
        Ax0_Core = self.Ax0_Core
        eri0_mo = self.eri0_mo
        T_iajb, t_iajb = self.T_iajb, self.t_iajb
        # If D_r is not an instance, we generate D_r using _get_D_r_oo_vv
        if self._D_r is NotImplemented:
            self._D_r = np.zeros((nmo, nmo))
            self._D_r[so, so] = - 2 * np.einsum("iakb, jakb -> ij", T_iajb, t_iajb)
            self._D_r[sv, sv] = 2 * np.einsum("iajc, ibjc -> ab", T_iajb, t_iajb)
        D_r = self._D_r

        L = np.zeros((nvir, nocc))
        L += Ax0_Core(sv, so, sa, sa)(D_r)
        L -= 4 * np.einsum("jakb, ijbk -> ai", T_iajb, eri0_mo[so, so, sv, so])
        L += 4 * np.einsum("ibjc, abjc -> ai", T_iajb, eri0_mo[sv, sv, so, sv])

        return L

    def _get_D_r(self):
        L = self.L
        D_r = self._D_r
        so, sv, sa = self.so, self.sv, self.sa
        Ax0_Core = self.Ax0_Core
        e, mo_occ = self.e, self.mo_occ
        D_r[sv, so] = cphf.solve(Ax0_Core(sv, so, sv, so), e, mo_occ, L, max_cycle=100, tol=1e-15)[0]

        conv = (
            + D_r[sv, so] * (self.ev[:, None] - self.eo[None, :])
            + Ax0_Core(sv, so, sv, so)(D_r[sv, so]) + L
        )
        print("Converge conv: ", abs(conv).max())
        if abs(conv).max() > 1e-8:
            msg = "\nget_E_1: CP-HF not converged well!\nMaximum deviation: " + str(abs(conv).max())
            warnings.warn(msg)
        return D_r

    def _get_W_I(self):
        so, sv = self.so, self.sv
        nmo = self.nmo
        T_iajb = self.T_iajb
        eri0_mo = self.eri0_mo
        W_I = np.zeros((nmo, nmo))
        W_I[so, so] = - 2 * np.einsum("iakb, jakb -> ij", T_iajb, eri0_mo[so, sv, so, sv])
        W_I[sv, sv] = - 2 * np.einsum("iajc, ibjc -> ab", T_iajb, eri0_mo[so, sv, so, sv])
        W_I[sv, so] = - 4 * np.einsum("jakb, ijbk -> ai", T_iajb, eri0_mo[so, so, sv, so])

        return W_I

    def _get_D_iajb(self):
        return (
            + self.eo[:, None, None, None]
            - self.ev[None, :, None, None]
            + self.eo[None, None, :, None]
            - self.ev[None, None, None, :]
        )

    # endregion
