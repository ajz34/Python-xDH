import numpy as np
from functools import partial
import os

from pyxdhalpha.DerivOnce.deriv_once import DerivOnce, DerivOnceNCDFT
from pyxdhalpha.Utilities import GridIterator, KernelHelper


MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class DipoleSCF(DerivOnce):

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError("This is still under construction...")

    def _get_H_1_ao(self):
        return - self.mol.intor("int1e_r")

    def _get_F_1_ao(self):
        return self.H_1_ao

    def _get_S_1_ao(self):
        return 0

    def _get_eri1_ao(self):
        return 0

    def _get_E_1(self):
        mol = self.mol
        H_1_ao = self.H_1_ao
        D = self.D

        dip_elec = np.einsum("Apq, pq -> A", H_1_ao, D)
        dip_nuc = np.einsum("A, At -> t", mol.atom_charges(), mol.atom_coords())

        return dip_elec + dip_nuc


class DipoleNCDFT(DerivOnceNCDFT, DipoleSCF):

    def Ax1_Core(self, si, sj, sk, sl):
        raise NotImplementedError

    @property
    def DerivOnceMethod(self):
        return DipoleSCF

    def _get_E_1(self):
        so, sv, sa = self.so, self.sv, self.sa
        B_1 = self.B_1
        Z = self.Z
        E_1 = 4 * np.einsum("ai, Aai -> A", Z, B_1[:, sv, so])
        E_1 += self.nc_deriv.E_1
        return E_1


class Test_DipoleSCF:

    def test_HF_dipole(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2()
        helper = DipoleSCF(H2O2.hf_eng)

        assert(np.allclose(
            helper.E_1, helper.scf_eng.dip_moment(unit="A.U."),
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-HF-freq.fchk"))

        assert(np.allclose(
            helper.E_1, formchk.dipole(),
            atol=1e-6, rtol=1e-4
        ))

    def test_B3LYP_dipole(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface

        H2O2 = Mol_H2O2()
        helper = DipoleSCF(H2O2.gga_eng)

        assert(np.allclose(
            helper.E_1, helper.scf_eng.dip_moment(unit="A.U."),
            atol=1e-6, rtol=1e-4
        ))

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-B3LYP-freq.fchk"))

        assert(np.allclose(
            helper.E_1, formchk.dipole(),
            atol=1e-6, rtol=1e-4
        ))

    def test_HF_B3LYP_dipole(self):

        import pickle
        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2

        H2O2 = Mol_H2O2()
        helper = DipoleNCDFT(H2O2.hf_eng, H2O2.gga_eng)
        with open(
                resource_filename("pyxdhalpha", "Validation/numerical_deriv/ncdft_derivonce_hf_b3lyp.dat"), "rb"
        ) as f:
            ref_grad = pickle.load(f)["dipole"]

        assert (np.allclose(
            helper.E_1, ref_grad,
            atol=1e-6, rtol=1e-4
        ))
