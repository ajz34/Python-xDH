from pyscf import gto, lib
from berny import Berny, geomlib


class OptimizeHelper:

    def __init__(self, mol_origin: gto.Mole):
        self.mol_origin = mol_origin
        self.mol_optimized = None
        self.gradientmax = 0.00045
        self.gradientrms = 0.00015
        self.stepmax = 0.0018
        self.steprms = 0.0012

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
        optimizer = Berny(self.mol_to_geom(mol), verbosity=-2,
                          gradientmax=self.gradientmax,
                          gradientrms=self.gradientrms,
                          stepmax=self.stepmax,
                          steprms=self.steprms)

        mol_opt = mol.copy()
        for geom in optimizer:
            energy, gradients = solver(mol_opt)
            optimizer.send((energy, gradients))
            mol_opt = self.geom_to_mol(mol, geom)
        return mol_opt
