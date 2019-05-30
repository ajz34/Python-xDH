import numpy as np
from functools import partial
import os

from pyxdhalpha.DerivOnce import DerivOnceMP2, GradSCF

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GradMP2(GradSCF, DerivOnceMP2):

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError("This is still under construction...")

    def _get_E_1(self):
        so, sv = self.so, self.sv
        natm = self.natm
        E_1 = (
            + np.einsum("pq, Apq -> A", self.D_r, self.B_1)
            + np.einsum("pq, Apq -> A", self.W_I, self.S_1_mo)
            + 2 * np.einsum("iajb, Aiajb -> A", self.T_iajb, self.eri1_mo[:, so, sv, so, sv])
        ).reshape(natm, 3)
        E_1 += super(GradMP2, self)._get_E_1()
        return E_1


class Test_GradMP2:

    def test_MP2_grad(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        from pyscf import mp, grad

        H2O2 = Mol_H2O2()
        gmh = GradMP2(H2O2.hf_eng)

        mp2_eng = mp.MP2(gmh.scf_eng)
        mp2_eng.kernel()
        mp2_grad = grad.mp2.Gradients(mp2_eng)
        mp2_grad.kernel()

        assert(np.allclose(
            gmh.E_1, mp2_grad.de,
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-MP2-freq.fchk"))

        assert(np.allclose(
            gmh.E_1, formchk.grad(),
            atol=1e-6, rtol=1e-4
        ))
