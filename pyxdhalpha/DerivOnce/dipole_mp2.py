import numpy as np
from functools import partial
import os

from pyxdhalpha.DerivOnce import DerivOnceMP2, DerivOnceXDH, DipoleSCF, DipoleNCDFT

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DipoleMP2(DerivOnceMP2, DipoleSCF):

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError("This is still under construction...")

    def _get_E_1(self):
        E_1 = np.einsum("pq, Apq -> A", self.D_r, self.B_1)
        E_1 += super(DipoleMP2, self)._get_E_1()
        return E_1


class DipoleXDH(DerivOnceXDH, DipoleSCF):

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError("This is still under construction...")

    @property
    def DerivOnceNCGGAClass(self):
        return DipoleNCDFT

    def _get_E_1(self):
        E_1 = np.einsum("pq, Apq -> A", self.D_r, self.B_1)
        E_1 += super(self.DerivOnceNCGGAClass, self.nc_deriv)._get_E_1()
        return E_1


class Test_DipoleMP2:

    def test_MP2_dipole(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2()
        dmh = DipoleMP2(H2O2.hf_eng)

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-MP2-freq.fchk"))

        assert(np.allclose(
            dmh.E_1, formchk.dipole(),
            atol=1e-6, rtol=1e-4
        ))

    def test_B2PLYP_dipole(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        dmh = DipoleMP2(H2O2.gga_eng, cc=0.27)

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-B2PLYP-freq.fchk"))

        assert(np.allclose(
            dmh.E_1, formchk.dipole(),
            atol=1e-6, rtol=1e-4
        ))

    def test_XYG3_dipole(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2_sc = Mol_H2O2(xc="B3LYPg")
        H2O2_nc = Mol_H2O2(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
        dmh = DipoleXDH(H2O2_sc.gga_eng, H2O2_nc.gga_eng, cc=0.3211)

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-XYG3-force.fchk"))

        assert(np.allclose(
            dmh.E_1, formchk.dipole(),
            atol=1e-6, rtol=1e-4
        ))
