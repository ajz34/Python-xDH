from pyscf import dft, gto
import pyscf.dft.numint
import numpy as np
from functools import partial
import os

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GridIterator:

    def __init__(self, mol, grids, D, deriv=3, memory=2000):

        self.mol = mol  # type: gto.Mole
        self.grids = grids  # type: dft.Grids
        self.D = D
        self.ni = dft.numint.NumInt()
        self.batch = self.ni.block_loop(mol, grids, mol.nao, deriv, self.mol.max_memory)

        self._ao = None
        self._ngrid = None
        self._weight = None
        self._ao_0 = None
        self._ao_1 = None
        self._ao_2 = None
        self._ao_2T = None
        self._ao_3 = None
        self._ao_3T = None
        self._rho_01 = None
        self._rho_0 = None
        self._rho_1 = None
        self._rho_2 = None
        self._A_rho_1 = None
        self._A_rho_2 = None
        self._A_gamma_1 = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.clear()
            self._ao, _, self._weight, _ = next(self.batch)
            return self
        except StopIteration:
            raise StopIteration

    def clear(self):
        self._ao = None
        self._ngrid = None
        self._weight = None
        self._ao_0 = None
        self._ao_1 = None
        self._ao_2 = None
        self._ao_2T = None
        self._ao_3T = None
        self._rho_01 = None
        self._rho_0 = None
        self._rho_1 = None
        self._rho_2 = None
        self._A_rho_1 = None
        self._A_rho_2 = None
        self._A_gamma_1 = None
        return

    @property
    def ngrid(self):
        return self.weight.size

    @property
    def weight(self):
        return self._weight

    @property
    def ao(self):
        return self._ao

    @property
    def ao_0(self):
        if self._ao_0 is None:
            self._ao_0 = self.ao[0]
        return self._ao_0

    @property
    def ao_1(self):
        if self._ao_1 is None:
            self._ao_1 = self.ao[1:4]
        return self._ao_1

    @property
    def ao_2T(self):
        if self._ao_2T is None:
            self._ao_2T = self.ao[4:10]
        return self._ao_2T

    @property
    def ao_2(self):
        if self._ao_2 is None:
            XX, XY, XZ, YY, YZ, ZZ = range(4, 10)
            ao = self.ao
            self._ao_2 = np.array([
                [ao[XX], ao[XY], ao[XZ]],
                [ao[XY], ao[YY], ao[YZ]],
                [ao[XZ], ao[YZ], ao[ZZ]],
            ])
        return self._ao_2

    @property
    def ao_3(self):
        if self._ao_3 is None:
            XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ = range(10, 20)
            ao = self.ao
            self._ao_3 = np.array([
                [[ao[XXX], ao[XXY], ao[XXZ]],
                 [ao[XXY], ao[XYY], ao[XYZ]],
                 [ao[XXZ], ao[XYZ], ao[XZZ]]],
                [[ao[XXY], ao[XYY], ao[XYZ]],
                 [ao[XYY], ao[YYY], ao[YYZ]],
                 [ao[XYZ], ao[YYZ], ao[YZZ]]],
                [[ao[XXZ], ao[XYZ], ao[XZZ]],
                 [ao[XYZ], ao[YYZ], ao[YZZ]],
                 [ao[XZZ], ao[YZZ], ao[ZZZ]]],
            ])
        return self._ao_3

    @property
    def ao_3T(self):
        if self._ao_3T is None:
            XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ = range(10, 20)
            ao = self.ao
            self._ao_3T = np.array([
                [ao[XXX], ao[XXY], ao[XXZ], ao[XYY], ao[XYZ], ao[XZZ]],
                [ao[XXY], ao[XYY], ao[XYZ], ao[YYY], ao[YYZ], ao[YZZ]],
                [ao[XXZ], ao[XYZ], ao[XZZ], ao[YYZ], ao[YZZ], ao[ZZZ]],
            ])
        return self._ao_3T

    @property
    def rho_01(self):
        if self._rho_01 is None:
            self._rho_01 = np.einsum("uv, rgu, gv -> rg", self.D, self.ao[0:4], self.ao_0)
            self._rho_01[1:4] *= 2
        return self._rho_01

    @property
    def rho_0(self):
        if self._rho_0 is None:
            self._rho_0 = self.rho_01[0]
        return self._rho_0

    @property
    def rho_1(self):
        if self._rho_1 is None:
            self._rho_1 = self.rho_01[1:4]
        return self._rho_1

    @property
    def rho_2(self):
        if self._rho_2 is None:
            self._rho_2 = (
                + 2 * np.einsum("uv, rgu, wgv -> rwg", self.D, self.ao_1, self.ao_1)
                + 2 * np.einsum("uv, rwgu, gv -> rwg", self.D, self.ao_2, self.ao_0)
            )
        return self._rho_2

    @property
    def A_rho_1(self):
        if self._A_rho_1 is None:
            natm = self.mol.natm
            self._A_rho_1 = np.zeros((natm, 3, self.ao.shape[1]))
            for A in range(natm):
                _, _, p0, p1 = self.mol.aoslice_by_atom()[A]
                sA = slice(p0, p1)
                self._A_rho_1[A] = - 2 * np.einsum("tgk, gl, kl -> tg ", self.ao_1[:, :, sA], self.ao_0, self.D[sA])
        return self._A_rho_1

    @property
    def A_rho_2(self):
        if self._A_rho_2 is None:
            natm = self.mol.natm
            self._A_rho_2 = np.zeros((natm, 3, 3, self.ao.shape[1]))
            for A in range(natm):
                _, _, p0, p1 = self.mol.aoslice_by_atom()[A]
                sA = slice(p0, p1)
                self._A_rho_2[A] = - 2 * np.einsum("trgk, gl, kl -> trg", self.ao_2[:, :, :, sA], self.ao_0, self.D[sA])
                self._A_rho_2[A] += - 2 * np.einsum("tgk, rgl, kl -> trg", self.ao_1[:, :, sA], self.ao_1, self.D[sA])
        return self._A_rho_2

    @property
    def A_gamma_1(self):
        if self._A_gamma_1 is None:
            self._A_gamma_1 = np.einsum("rg, Atrg -> Atg", self.rho_1, self.A_rho_2)
        return self._A_gamma_1
