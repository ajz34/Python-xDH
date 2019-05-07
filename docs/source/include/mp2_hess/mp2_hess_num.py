from pyxdh.utilities import NumericDiff
from pyscf import scf, gto, grad, mp
import pickle
import numpy as np

np.set_printoptions(8, linewidth=1000, suppress=True)


def mol_to_mp2grad(mol):
    scf_eng = scf.RHF(mol)
    scf_eng.conv_tol_grad = 1e-9
    scf_eng.max_cycle = 400
    scf_eng.kernel()
    mp2_eng = mp.RMP2(scf_eng)
    mp2_eng.kernel()
    mp2_grad = grad.mp2.Gradients(mp2_eng)
    grad_mp2 = mp2_grad.kernel()
    return grad_mp2


def main():
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

    grad_mp2_diff = NumericDiff(mol, mol_to_mp2grad, deriv=2).get_numdif()
    d = {
        "grad_mp2_diff": grad_mp2_diff,
    }
    with open("mp2_hess_num.dat", "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    print(grad_mp2_diff)


if __name__ == '__main__':
    main()
