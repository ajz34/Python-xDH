from pyscf import gto
from hf_helper import HFHelper
import numpy as np


mol = gto.Mole()
mol.atom = """
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.5  0.0  0.0
H  0.0  0.7  1.5
"""
mol.basis = "6-31G"
mol.verbose = 0
mol.build()

nmo = mol.nao
nocc = mol.nelec[0]
so = slice(0, nocc)
sa = slice(0, nmo)

scfh = HFHelper(mol)


class TestHFHelper(object):
    def test_H_1_ao(self):
        assert True


def test_return_true():
    assert 2 == 3
