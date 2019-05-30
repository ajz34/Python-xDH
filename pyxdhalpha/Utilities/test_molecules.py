from pyscf import scf, gto, grad, dft, hessian


class Mol_H2O2:

    def __init__(self):
        mol = gto.Mole()
        mol.atom = """
        O  0.0  0.0  0.0
        O  0.0  0.0  1.5
        H  1.0  0.0  0.0
        H  0.0  0.7  1.0
        """
        mol.basis = "6-31G"
        mol.verbose = 0
        mol.build()

        self.mol = mol

        self._hf_eng = NotImplemented
        self._hf_grad = NotImplemented
        self._gga_eng = NotImplemented
        self._gga_grad = NotImplemented
        
    @property
    def hf_eng(self):
        if self._hf_eng is not NotImplemented:
            return self._hf_eng
        hf_eng = scf.RHF(self.mol)
        hf_eng.kernel()
        self._hf_eng = hf_eng
        return self._hf_eng
    
    @property
    def hf_grad(self):
        if self._hf_grad is not NotImplemented:
            return self._hf_grad
        hf_grad = grad.RHF(self.hf_eng)
        hf_grad.kernel()
        self._hf_grad = hf_grad
        return self._hf_grad

    @property
    def gga_eng(self):
        if self._gga_eng is not NotImplemented:
            return self._gga_eng

        grids = dft.Grids(self.mol)
        grids.atom_grid = (99, 590)
        grids.becke_scheme = dft.gen_grid.stratmann
        grids.build()

        gga_eng = scf.RKS(self.mol)
        gga_eng.grids = grids
        gga_eng.conv_tol = 1e-11
        gga_eng.conv_tol_grad = 1e-9
        gga_eng.max_cycle = 100
        gga_eng.xc = "B3LYPg"
        gga_eng.kernel()

        self._gga_eng = gga_eng
        return self._gga_eng

    @property
    def gga_grad(self):
        if self._gga_grad is not NotImplemented:
            return self._gga_grad
        gga_grad = grad.RKS(self.gga_eng)
        gga_grad.kernel()
        self._gga_grad = gga_grad
        return self._gga_grad

