import numpy as np
from abc import ABC, abstractmethod
from functools import partial
import os
import warnings

from pyscf import gto, dft, grad, hessian, lib
import pyscf.dft.numint
from pyscf.scf import cphf

from pyxdhalpha.DerivOnce import DerivOnceSCF, DerivOnceNCDFT
from pyxdhalpha.Utilities import timing, GridIterator, KernelHelper

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


# Cubic Inheritance: C1
class DerivTwiceSCF(ABC):

    def __init__(self, config):

        # From configuration file, with default values
        self.config = config  # type: dict
        self.A = config["deriv_A"]  # type: DerivOnceSCF
        self.B = config["deriv_B"]  # type: DerivOnceSCF
        self.deriv_same = config["deriv_same"]  # type: bool
        self.grdit_memory = 2000
        if "grdit_memory" in config:
            self.grdit_memory = config["grdit_memory"]

        # Make assertion on coefficient idential of deriv_A and deriv_B instances
        # for some molecules which have degenerate orbital energies,
        # two instances of DerivOnce have different coefficients can be fatal
        assert(np.allclose(self.A.C, self.B.C))
        # After assertion passed, then we can say things may work; however we should not detect intended sabotage
        # So it is recommended to initialize deriv_A and deriv_B with the same runned scf.RHF instance

        # Basic Information
        self._mol = self.A.mol
        self._C = self.A.C
        self._e = self.A.e
        self._D = self.A.D
        self._mo_occ = self.A.mo_occ

        self.cx = self.A.cx
        self.xc = self.A.xc
        self.grids = self.A.grids
        self.xc_type = self.A.xc_type

        # Matrices
        self._H_2_ao = NotImplemented
        self._H_2_mo = NotImplemented
        self._S_2_ao = NotImplemented
        self._S_2_mo = NotImplemented
        self._F_2_ao = NotImplemented
        self._F_2_ao_Jcontrib = NotImplemented
        self._F_2_ao_Kcontrib = NotImplemented
        self._F_2_ao_GGAcontrib = NotImplemented
        self._F_2_mo = NotImplemented
        self._Xi_2 = NotImplemented
        self._B_2 = NotImplemented

        # E_2
        self._E_2_SS = NotImplemented
        self._E_2_SU = NotImplemented
        self._E_2_US = NotImplemented
        self._E_2_UU = NotImplemented
        self._E_2 = NotImplemented

    # region Properties

    @property
    def mol(self):
        return self._mol

    @property
    def C(self):
        return self._C

    @property
    def Co(self):
        return self.C[:, self.so]

    @property
    def Cv(self):
        return self.C[:, self.sv]

    @property
    def nmo(self):
        return self.C.shape[1]

    @property
    def nao(self):
        return self.mol.nao

    @property
    def nocc(self):
        return self.mol.nelec[0]

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def mo_occ(self):
        return self._mo_occ

    @property
    def natm(self):
        return self.mol.natm

    @property
    def sa(self):
        return slice(0, self.nmo)

    @property
    def so(self):
        return slice(0, self.nocc)

    @property
    def sv(self):
        return slice(self.nocc, self.nmo)

    @property
    def e(self):
        return self._e

    @property
    def eo(self):
        return self.e[self.so]

    @property
    def ev(self):
        return self.e[self.sv]

    @property
    def D(self):
        return self._D

    @property
    def H_2_ao(self):
        if self._H_2_ao is NotImplemented:
            self._H_2_ao = self._get_H_2_ao()
        return self._H_2_ao

    @property
    def H_2_mo(self):
        if self._H_2_mo is NotImplemented:
            self._H_2_mo = self._get_H_2_mo()
        return self._H_2_mo

    @property
    def S_2_ao(self):
        if self._S_2_ao is NotImplemented:
            self._S_2_ao = self._get_S_2_ao()
        return self._S_2_ao

    @property
    def S_2_mo(self):
        if self._S_2_mo is NotImplemented:
            self._S_2_mo = self._get_S_2_mo()
        return self._S_2_mo

    @property
    def F_2_ao_Jcontrib(self):
        if self._F_2_ao_Jcontrib is NotImplemented:
            self._F_2_ao_Jcontrib = self._get_F_2_ao_Jcontrib()
        return self._F_2_ao_Jcontrib

    @property
    def F_2_ao_Kcontrib(self):
        if self._F_2_ao_Kcontrib is NotImplemented:
            self._F_2_ao_Kcontrib = self._get_F_2_ao_Kcontrib()
        return self._F_2_ao_Kcontrib

    @property
    def F_2_ao_GGAcontrib(self):
        if self._F_2_ao_GGAcontrib is NotImplemented:
            self._F_2_ao_GGAcontrib = self._get_F_2_ao_GGAcontrib()
        return self._F_2_ao_GGAcontrib

    @property
    def F_2_ao(self):
        if self._F_2_ao is NotImplemented:
            self._F_2_ao = self._get_F_2_ao()
        return self._F_2_ao

    @property
    def F_2_mo(self):
        if self._F_2_mo is NotImplemented:
            self._F_2_mo = self._get_F_2_mo()
        return self._F_2_mo

    @property
    def Xi_2(self):
        if self._Xi_2 is NotImplemented:
            self._Xi_2 = self._get_Xi_2()
        return self._Xi_2

    @property
    def B_2(self):
        if self._B_2 is NotImplemented:
            self._B_2 = self._get_B_2()
        return self._B_2

    # endregion

    # region Getters

    @abstractmethod
    def _get_H_2_ao(self):
        pass

    def _get_H_2_mo(self):
        return self.C.T @ self.H_2_ao @ self.C

    @abstractmethod
    def _get_S_2_ao(self):
        pass

    def _get_S_2_mo(self):
        return self.C.T @ self.S_2_ao @ self.C

    @abstractmethod
    def _get_F_2_ao_Jcontrib(self):
        pass

    @abstractmethod
    def _get_F_2_ao_Kcontrib(self):
        pass

    @abstractmethod
    def _get_F_2_ao_GGAcontrib(self):
        pass

    def _get_F_2_ao(self):
        return self.H_2_ao + self.F_2_ao_Jcontrib - 0.5 * self.cx * self.F_2_ao_Kcontrib + self.F_2_ao_GGAcontrib

    def _get_F_2_mo(self):
        return self.C.T @ self.F_2_ao @ self.C

    def _get_Xi_2(self):
        A = self.A
        B = self.B

        Xi_2 = (
            self.S_2_mo
            + np.einsum("Apm, Bqm -> ABpq", A.U_1, B.U_1)
            + np.einsum("Bpm, Aqm -> ABpq", B.U_1, A.U_1)
            - np.einsum("Apm, Bqm -> ABpq", A.S_1_mo, B.S_1_mo)
            - np.einsum("Bpm, Aqm -> ABpq", B.S_1_mo, A.S_1_mo)
        )
        return Xi_2

    def _get_B_2(self):
        A = self.A
        B = self.B
        Ax0_Core = A.Ax0_Core  # Ax0_Core should be the same for A and B derivative

        sa, so, sv = self.sa, self.so, self.sv
        e = self.e

        B_2 = (
            # line 1
            + self.F_2_mo
            - np.einsum("ABai, i -> ABai", self.Xi_2, e)
            - 0.5 * Ax0_Core(sa, sa, so, so)(self.Xi_2[:, :, :, :, so, so])
            # line 2
            + np.einsum("Apa, Bpi -> ABai", A.U_1, B.F_1_mo)
            + np.einsum("Api, Bpa -> ABai", A.U_1, B.F_1_mo)
            + np.einsum("Bpa, Bpi -> ABai", B.U_1, A.F_1_mo)
            + np.einsum("Bpi, Bpa -> ABai", B.U_1, A.F_1_mo)
            # line 3
            + np.einsum("Apa, Bpi, p -> ABai", A.U_1, B.U_1, e)
            + np.einsum("Bpa, Api, p -> ABai", B.U_1, A.U_1, e)
            # line 4
            + 0.5 * Ax0_Core(sa, sa, sa, sa)(
                + np.einsum("Akm, Blm -> ABkl", A.U_1[:, :, :, so], B.U_1[:, :, :, so])
                + np.einsum("Bkm, Alm -> ABkl", B.U_1[:, :, :, so], A.U_1[:, :, :, so])
            )
            # line 5
            + np.einsum("Apa, Bpi -> ABai", A.U_1, Ax0_Core(sa, sa, sa, so)(B.U_1[:, :, :, so]))
            + np.einsum("Bpa, Api -> ABai", B.U_1, Ax0_Core(sa, sa, sa, so)(A.U_1[:, :, :, so]))
            # line 6
            + np.einsum("Api, Bpa -> ABai", A.U_1, Ax0_Core(sa, sa, sa, so)(B.U_1[:, :, :, so]))
            + np.einsum("Bpi, Apa -> ABai", B.U_1, Ax0_Core(sa, sa, sa, so)(A.U_1[:, :, :, so]))
            # line 7
            + A.Ax1_Core(sa, sa, sa, so)(B.U_1[:, :, :, so])
            + B.Ax1_Core(sa, sa, sa, so)(A.U_1[:, :, :, so]).swapaxes(0, 1)
        )
        return B_2

    # endregion
