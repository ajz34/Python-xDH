import numpy as np

from pyxdhalpha.DerivTwice import DerivTwiceMP2, DerivTwiceXDH, HessSCF, HessNCDFT


# Cubic Inheritance: C2
class HessMP2(DerivTwiceMP2, HessSCF):
    pass


class HessXDH(DerivTwiceXDH, HessMP2, HessNCDFT):

    def _get_E_2_Skeleton(self, grids=None, xc=None, cx=None, xc_type=None):
        return HessNCDFT._get_E_2_Skeleton(self, grids, xc, cx, xc_type)


class Test_HessMP2:

    def test_MP2_hess(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface
        from pyxdhalpha.DerivOnce import GradMP2

        H2O2 = Mol_H2O2()
        config = {
            "scf_eng": H2O2.hf_eng,
            "rotation": True,
        }
        grad_helper = GradMP2(config)
        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
        }

        helper = HessMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-MP2-freq.fchk"))

        assert(np.allclose(
            E_2, formchk.hessian(),
            atol=1e-6, rtol=1e-4
        ))

    def test_B2PLYP_hess(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.Utilities import FormchkInterface
        from pyxdhalpha.DerivOnce import GradMP2
        import pickle

        H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
        config = {
            "scf_eng": H2O2.gga_eng,
            "cc": 0.27
        }
        grad_helper = GradMP2(config)

        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
        }

        helper = HessMP2(config)
        E_2 = helper.E_2

        formchk = FormchkInterface(resource_filename("pyxdhalpha", "Validation/gaussian/H2O2-B2PLYP-freq.fchk"))

        assert(np.allclose(
            E_2, formchk.hessian(),
            atol=1e-5, rtol=1e-4
        ))

        with open(resource_filename("pyxdhalpha", "Validation/numerical_deriv/mp2_hessian_b2plyp.dat"), "rb") as f:
            ref_hess = pickle.load(f)["hess"]

        assert (np.allclose(
            E_2, ref_hess,
            atol=1e-6, rtol=1e-4
        ))

    def test_XYG3_hess(self):

        from pkg_resources import resource_filename
        from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
        from pyxdhalpha.DerivOnce import GradXDH
        from pyxdhalpha.DerivTwice import HessXDH
        import pickle

        H2O2_sc = Mol_H2O2(xc="B3LYPg")
        H2O2_nc = Mol_H2O2(xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
        config = {
            "scf_eng": H2O2_sc.gga_eng,
            "nc_eng": H2O2_nc.gga_eng,
            "cc": 0.3211,
        }
        grad_helper = GradXDH(config)

        config = {
            "deriv_A": grad_helper,
            "deriv_B": grad_helper,
        }
        helper = HessXDH(config)
        E_2 = helper.E_2

        with open(resource_filename("pyxdhalpha", "Validation/numerical_deriv/xdh_hessian_xyg3.dat"), "rb") as f:
            ref_hess = pickle.load(f)["hess"]

        assert (np.allclose(
            E_2, ref_hess,
            atol=1e-6, rtol=1e-4
        ))
