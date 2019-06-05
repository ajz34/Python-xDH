import numpy as np
from abc import ABC
from functools import partial
import os
import warnings

from pyscf.scf import cphf

from pyxdhalpha.DerivTwice import DerivTwiceSCF
from pyxdhalpha.DerivOnce import DerivOnceMP2

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


# Cubic Inheritance: C1
class DerivTwiceMP2(DerivTwiceSCF, ABC):

    def __init__(self, config):
        super(DerivTwiceMP2, self).__init__(config)
        # Only make IDE know these two instances are DerivOnceMP2 classes
        self.A = config["deriv_A"]  # type: DerivOnceMP2
        self.B = config["deriv_B"]  # type: DerivOnceMP2
        assert(isinstance(self.A, DerivOnceMP2))
        assert(isinstance(self.B, DerivOnceMP2))
        assert(self.A.cc == self.B.cc)
        self.cc = self.A.cc

        # For simplicity, these values are not set to be properties
        # However, these values should not be changed or redefined
        self.t_iajb = self.A.t_iajb
        self.T_iajb = self.A.T_iajb
        self.L = self.A.L
        self.D_r = self.A.D_r
        self.W_I = self.A.W_I
        self.D_iajb = self.A.D_iajb

    # region Properties

    # endregion

    # region Functions

    def _get_E_2_MP2_Contrib1(self):
        return np.einsum("pq, ABpq -> AB", self.D_r, self.pdB_F_A_mo)

    # endregion


if __name__ == '__main__':

    from pkg_resources import resource_filename
    from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
    from pyxdhalpha.Utilities import FormchkInterface
    from pyxdhalpha.DerivOnce import GradSCF

    H2O2 = Mol_H2O2()
    config = {
        "scf_eng": H2O2.hf_eng
    }
    helper = GradSCF(config)
    hf_grad = helper.scf_grad
    print(helper.pdA_F_0_mo)

    so, sv, sa = helper.so, helper.sv, helper.sa
    U_1 = helper.U_1
