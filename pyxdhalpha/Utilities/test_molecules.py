from pyscf import scf, gto, grad, dft, hessian


class Mol_H2O2:

    def __init__(self):
        mol = gto.Mole()
        mol.atom = """
        O  0.0  0.0  0.0
        O  0.0  0.0  1.5
        H  1.0  0.0  0.0
        H  0.0  1.0  0.7
        """
        mol.basis = "6-31G"
        mol.verbose = 0
        mol.build()
        hf_eng = scf.RHF(mol)
        hf_eng.kernel()
        hf_grad = grad.RHF(hf_eng)
        hf_grad.kernel()

        self.mol = mol
        self.hf_eng = hf_eng
        self.hf_grad = hf_grad
