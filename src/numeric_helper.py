from pyscf import gto, lib
import numpy as np


# BOHR = 0.52917721092
BOHR = lib.param.BOHR


class NumericDiff:

    def __init__(self, mol, met, p5=False, idx=None, interval=3e-4, deriv=1, symm=True):
        """
        Numeric difference for quantum chemistry tensors.

        To use this class, you need at least provide a molecule object, and the
        analytical value generation method to be derivated.

        Take care for option ``deriv``. If you are to calculate Hessian of a
        value or tensor, ``deriv`` should be set to 2.

        Parameters
        ----------
        mol : pyscf.gto.Mole
            Molecule with base coordinate.
        met : function
            This function should return np.ndarray object
        p5 : bool, optional, default: False
            Option of utilizing 5-point stencil
        idx : int or None, optional, default: None
            Tensor dimension to be differenced;
            For example:
            Energy derivate and Gradient derivative should be given 0.
            ERI derivative should be given 4.
            If not given, program would check this value.
        interval : int, optional, default: 3e-4
            Difference interval. Not always better when smaller.
        deriv : {1, 2}, optional, default : 1
            Derivative order. For example:
            Energy derivative, or Fock matrix derivative, should be given 1;
            Gradient derivative, or derivative of analytic Hamiltonian core
            derivative, should be given 2.
        symm : bool, optional, default: True
            Judge whether symmtrize 2nd derivative.
            Symmtrize means (ABts) -> 0.5 * (ABts) + 0.5 (BAst).
            Since most tensors are symmtrized in this way, this option is True
            by default.
        """
        self.mol = mol  # type: gto.Mole
        self.met = met
        self.metsp = met(mol).shape

        self.p5 = p5
        self.interval = interval
        self.deriv = deriv
        self.symm = symm
        if idx is None:
            self.idx = len(self.metsp) - 2 * (deriv - 1)
        else:
            self.idx = idx
        self.val = None
        self.numdif = None  # type: np.ndarray
        return

    def move_mol(self, movelist):
        """
        Make a temporary molecule, which is moved away from the original molecule
        according to a pre-defined move list.

        Parameters
        ----------
        movelist : list of (tuple of int)
            The list is a series of movements on the temporary molecule.
            Every tuple in a list should contain 3 values. 1st value refers to
            atom index, 2nd value coordinate index, 3rd value times of interval
            to be moved away from original molecule.

        Returns
        -------
        mol_ret : pyscf.gto.Mole
        """
        # -------
        mol_ret = self.mol.copy()  # type: gto.Mole
        mol_ret_coords = mol_ret.atom_coords() * BOHR
        for tup in movelist:
            mol_ret_coords[tup[0], tup[1]] += tup[2] * self.interval
        mol_ret.set_geom_(mol_ret_coords)
        return mol_ret

    def calc_met(self):
        """
        Calculate all variables essential for numerical differention, and store
        in ``self.val``.
        """
        natm = self.mol.natm
        valsp = list(self.metsp)
        # -------
        if self.p5:  # Need 4 points
            valsp.insert(len(valsp) - self.idx, 4)
        else:  # Need 2 points
            valsp.insert(len(valsp) - self.idx, 2)
        if self.deriv == 1:
            # The matrix we need is (A, t, h, ...)
            valsp.insert(0, natm)
            valsp.insert(1, 3)
        elif self.deriv == 2:
            # Assert when deriv=2, the shape passed in is like (natm_B, 3_s, h, ...)
            # The matrix we need is (B, A, s, t, h, ...)
            valsp.insert(1, natm)
            valsp.insert(3, 3)
            assert (valsp[0] == valsp[1])
            assert (valsp[2] == valsp[3])
        self.val = np.empty(valsp)
        looplist = [(A, t, h)
                    for A in range(natm)
                    for t in range(3)
                    for h in range(2 + self.p5 * 2)]
        if self.p5:
            dev_h = [-2, -1, 1, 2]
        else:
            dev_h = [-1, 1]
        for A, t, h in looplist:
            movelist = [(A, t, dev_h[h])]
            if self.deriv == 1:
                self.val[(A, t,) + (slice(None),) * (len(self.metsp) - self.idx) + (h,)] = self.met(
                    self.move_mol(movelist))
            elif self.deriv == 2:
                self.val[(slice(None), A, slice(None), t) + (slice(None),) * (len(self.metsp) - self.idx - 2) + (
                h,)] = self.met(self.move_mol(movelist))
        return

    def calc_numdiv(self):
        """
        Calculate derivative by numerical differention.
        """
        slice_base = (slice(None),) * (len(self.metsp) - self.idx + 2)
        if not self.p5:
            self.numdif = (self.val[slice_base + (1,)] - self.val[slice_base + (0,)]) / (2 * self.interval / BOHR)
        else:
            self.numdif = (
                self.val[slice_base + (0,)]
                - 8 * self.val[slice_base + (1,)]
                + 8 * self.val[slice_base + (2,)]
                - self.val[slice_base + (3,)]
            ) / (12 * self.interval / BOHR)
        if self.deriv == 2 and self.symm:
            self.numdif += self.numdif.swapaxes(0, 1).swapaxes(2, 3)
            self.numdif /= 2
        return

    def get_numdif(self):
        """
        All calculations can be done here.

        Returns
        -------
        self.numdif : ndarray
            Final derivative result
        """
        self.calc_met()
        self.calc_numdiv()
        return self.numdif

