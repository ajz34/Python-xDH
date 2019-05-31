import pickle
from pyscf import scf

from pyxdhalpha.Utilities.test_molecules import Mol_H2O2
from pyxdhalpha.Utilities import NumericDiff, NucCoordDerivGenerator, DipoleDerivGenerator
from pyxdhalpha.DerivOnce import GradNCDFT


def mol_to_grad_helper(mol):
    H2O2 = Mol_H2O2(mol=mol)
    H2O2.hf_eng.kernel()
    helper = GradNCDFT(H2O2.gga_eng, H2O2.hf_eng.coeff, H2O2.hf_eng.mo_energy)
    return helper


def dipole_generator(component, interval):
    H2O2 = Mol_H2O2()
    mf = GradNCDFT(H2O2.gga_eng, H2O2.hf_eng.coeff, H2O2.hf_eng.mo_energy).scf_eng
    mf.get_hcore = lambda mol: scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
    return mf.kernel()


if __name__ == '__main__':
    result_dict = {}
    H2O2 = Mol_H2O2()

    num_obj = NucCoordDerivGenerator(H2O2.mol, mol_to_grad_helper)
    num_dif = NumericDiff(num_obj, lambda helper: helper.eng)
    result_dict["grad"] = num_dif.derivative

    num_obj = DipoleDerivGenerator(dipole_generator)
    num_dif = NumericDiff(num_obj)
    result_dict["dipole"] = num_dif.derivative

    with open("ncdft_derivonce_hf_b3lyp.dat", "wb") as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
