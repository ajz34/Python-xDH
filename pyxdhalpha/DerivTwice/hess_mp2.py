import numpy as np
from abc import ABC, abstractmethod
from functools import partial
import os

from pyscf import gto, dft, grad, hessian, lib
import pyscf.dft.numint
from pyscf.scf import _vhf, cphf

from pyxdhalpha.DerivTwice import DerivTwiceSCF, DerivTwiceMP2, HessSCF
from pyxdhalpha.Utilities import timing, GridIterator, KernelHelper


# Cubic Inheritance: C2
class HessMP2(DerivTwiceMP2, HessSCF):

    pass


if __name__ == '__main__':

    from pkg_resources import resource_filename
    from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
    from pyxdhalpha.Utilities import FormchkInterface
    from pyxdhalpha.DerivOnce import GradMP2

    H2O2 = Mol_H2O2()
    config = {
        "scf_eng": H2O2.hf_eng,
        "rotation": True,
    }
    scf_helper = GradMP2(config)
    config = {
        "deriv_A": scf_helper,
        "deriv_B": scf_helper,
    }

    helper = HessMP2(config)

    print(helper.E_2)

    # B2PLYP

    H2O2 = Mol_H2O2(xc="0.53*HF + 0.47*B88, 0.73*LYP")
    config = {
        "scf_eng": H2O2.gga_eng,
        "cc": 0.27
    }
    scf_helper = GradMP2(config)
    print(scf_helper.E_1)

    config = {
        "deriv_A": scf_helper,
        "deriv_B": scf_helper,
    }
    helper = HessMP2(config)
    print(helper.E_2)
