from pyscf import gto, lib
from berny import Berny, geomlib


class OptimizeHelper:

    def __init__(self, mol_origin: gto.Mole):
        self.mol_origin = mol_origin
        self.mol_optimized = None
        self.gradientmax = 0.000450
        self.gradientrms = 0.000300
        self.stepmax = 0.001800
        self.steprms = 0.001200

    def set_verytight(self):
        self.gradientmax = 0.000002
        self.gradientrms = 0.000001
        self.stepmax = 0.000006
        self.steprms = 0.000004
        return self

    @staticmethod
    def mol_to_geom(mol):
        # almost idential to PySCF:
        #     `geomopt.berny_solver.to_berny_geom(mol)`
        species = [mol.atom_symbol(i) for i in range(mol.natm)]
        coords = mol.atom_coords() * lib.param.BOHR
        geom = geomlib.Geometry(species, coords)
        return geom

    @staticmethod
    def geom_to_mol(mol, geom):
        # almost idential to PySCF:
        #     `mol_ret = mol.copy().set_geom_(geomopt.berny_solver._geom_to_atom(mol, geom), unit='Bohr')`
        mol_ret = mol.copy()
        mol_ret.set_geom_(geom.coords)
        return mol_ret

    def optimize(self, solver):

        mol = self.mol_origin
        geom_outer = self.mol_to_geom(mol)
        optimizer = Berny(geom_outer, verbosity=10,
                          gradientmax=self.gradientmax,
                          gradientrms=self.gradientrms,
                          stepmax=self.stepmax,
                          steprms=self.steprms)

        for geom in optimizer:
            mol_opt = self.geom_to_mol(mol, geom)
            print("In optimization: Molecular geom")
            print(mol_opt.atom_coords() * lib.param.BOHR)
            energy, gradients = solver(mol_opt)
            optimizer.send((energy, gradients))
            geom_outer = geom
        return self.geom_to_mol(mol, geom_outer)


if __name__ == '__main__':
    from pyscf import dft
    from gga_helper import GGAHelper
    from ncgga_engine import NCGGAEngine
    from pyscf import geomopt
    from pyscf import grad
    import pyscf.grad.rks

    mol = gto.Mole()
    mol.atom = """
        O  0.0  0.0  0.0
        H  1.0  0.0  0.0
        H  0.0  1.0  0.0
        """
    mol.basis = "6-31G"
    mol.verbose = 0
    mol.build()

    def mol_to_grids(mol):
        grids = dft.gen_grid.Grids(mol)
        grids.atom_grid = (75, 302)
        grids.becke_scheme = dft.gen_grid.stratmann
        grids.build()
        return grids

    def mol_to_E_0_E_1(mol):
        grids = mol_to_grids(mol)
        scfh = GGAHelper(mol, "b3lypg", grids)
        nch = GGAHelper(mol, "b3lypg", grids, init_scf=False)
        ncgga = NCGGAEngine(scfh, nch)
        E_0 = ncgga.E_0
        E_1 = ncgga.E_1
        print("In Optimization: E_0 = {}".format(E_0))
        return E_0, E_1

    scf_eng = dft.RKS(mol)
    scf_eng.xc = "b3lypg"
    scf_eng.grids = mol_to_grids(mol)
    scf_grad = grad.rks.Gradients(scf_eng)

    mol_optimized = geomopt.optimize(scf_eng, verbose=4)
    print("-----\n", gto.mole.cart2zmat(mol_optimized.atom_coords()))
    mol_optimized = OptimizeHelper(mol).optimize(mol_to_E_0_E_1)
    print("-----\n", gto.mole.cart2zmat(mol_optimized.atom_coords()))
