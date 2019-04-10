from pyscf import gto, scf, grad, lib, hessian, dft
import pyscf.grad.rks
import pyscf.hessian.rks
import pyscf.dft.numint
from hf_helper import HFHelper
from grid_helper import GridHelper, KernelHelper
from grid_iterator import GridIterator
import numpy as np
from functools import partial
import os, warnings

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GGAHelper(HFHelper):

    def __init__(self, mol, xc, grids, init_scf=True):
        self.xc = xc  # type: str
        self.grids = grids  # type: dft.gen_grid.Grids
        self.scf_eng = None  # type: dft.rks.RKS
        self.scf_grad = None  # type: grad.rks.Gradients
        self.scf_hess = None  # type: hessian.rks.Hessian
        self.cx = None
        self.grdh = None  # type: GridHelper
        self.kerh = None  # type: KernelHelper
        super(GGAHelper, self).__init__(mol, init_scf=init_scf)
        return

    def initialization_pyscf(self):
        self.scf_eng = scf.RKS(self.mol)
        self.scf_eng.conv_tol = 1e-11
        self.scf_eng.conv_tol_grad = 1e-9
        self.scf_eng.max_cycle = 100
        self.scf_eng.xc = self.xc
        self.scf_eng.grids = self.grids
        if self.init_scf:
            self.eng = self.scf_eng.kernel()
            if not self.scf_eng.converged:
                warnings.warn("SCF not converged!")
        self.scf_grad = grad.rks.Gradients(self.scf_eng)
        self.scf_hess = hessian.rks.Hessian(self.scf_eng)
        self.cx = dft.numint.NumInt().hybrid_coeff(self.xc)
        return

    def get_grdh(self):
        self.grdh = GridHelper(self.mol, self.grids, self.D)
        return self.grdh

    def get_kerh(self):
        if self.grdh is None:
            self.get_grdh()
        self.kerh = KernelHelper(self.grdh, self.xc)
        return self.kerh

    def Ax0_Core_deprecate(self, si, sj, sk, sl, reshape=True):
        C = self.C
        if self.kerh is None:
            self.get_kerh()
        grdh = self.grdh
        kerh = self.kerh

        def fx(mo1_):
            mo1 = mo1_.copy()  # type: np.ndarray
            shape1 = list(mo1.shape)
            mo1.shape = (-1, shape1[-2], shape1[-1])
            dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.transpose(0, 2, 1)
            r = np.empty((mo1.shape[0], si.stop - si.start, sj.stop - sj.start))
            for idx, dmX in enumerate(dm1):
                tmp_K = np.einsum("kl, gl -> gk", dmX, grdh.ao_0)
                rho_X_0 = np.einsum("gk, gk -> g", grdh.ao_0, tmp_K)
                rho_X_1 = 2 * np.einsum("rgk, gk -> rg", grdh.ao_1, tmp_K)
                gamma_XD = np.einsum("rg, rg -> g", rho_X_1, grdh.rho_1)
                tmp_M = np.empty((4, grdh.ngrid))
                tmp_M[0] = (
                    np.einsum("g, g -> g", rho_X_0, kerh.frr)
                    + 2 * np.einsum("g, g -> g", gamma_XD, kerh.frg)
                )
                tmp_M[1:4] = (
                    + 4 * np.einsum("g, g, rg -> rg", rho_X_0, kerh.frg, grdh.rho_1)
                    + 8 * np.einsum("g, g, rg -> rg", gamma_XD, kerh.fgg, grdh.rho_1)
                    + 4 * np.einsum("rg, g -> rg", rho_X_1, kerh.fg)
                )
                ax_ao = (
                    + 1 * self.scf_eng.get_j(dm=dmX)
                    - 0.5 * self.cx * self.scf_eng.get_k(dm=dmX)
                    + np.einsum("rg, rgu, gv -> uv", tmp_M, grdh.ao[:4], grdh.ao_0)
                )
                ax_ao += ax_ao.T
                r[idx] = C[:, si].T @ ax_ao @ C[:, sj]

            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.append(r.shape[-2])
                shape1.append(r.shape[-1])
                r.shape = shape1
            return r

        return fx

    def Ax0_Core(self, si, sj, sk, sl, reshape=True):
        C = self.C
        nao = self.mol.nao

        def fx(mo1_):
            mo1 = mo1_.copy()  # type: np.ndarray
            shape1 = list(mo1.shape)
            mo1.shape = (-1, shape1[-2], shape1[-1])
            dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.transpose(0, 2, 1)
            ax_ao = np.empty((dm1.shape[0], nao, nao))
            for idx, dmX in enumerate(dm1):
                ax_ao[idx] = (
                    + 1 * self.scf_eng.get_j(dm=dmX)
                    - 0.5 * self.cx * self.scf_eng.get_k(dm=dmX)
                )
            grdit = GridIterator(self.mol, self.grids, self.D, deriv=2)
            for grdh in grdit:
                kerh = KernelHelper(grdh, self.xc)
                for idx, dmX in enumerate(dm1):
                    tmp_K = np.einsum("kl, gl -> gk", dmX, grdh.ao_0)
                    rho_X_0 = np.einsum("gk, gk -> g", grdh.ao_0, tmp_K)
                    rho_X_1 = 2 * np.einsum("rgk, gk -> rg", grdh.ao_1, tmp_K)
                    gamma_XD = np.einsum("rg, rg -> g", rho_X_1, grdh.rho_1)
                    tmp_M = np.empty((4, grdh.ngrid))
                    tmp_M[0] = (
                        np.einsum("g, g -> g", rho_X_0, kerh.frr)
                        + 2 * np.einsum("g, g -> g", gamma_XD, kerh.frg)
                    )
                    tmp_M[1:4] = (
                        + 4 * np.einsum("g, g, rg -> rg", rho_X_0, kerh.frg, grdh.rho_1)
                        + 8 * np.einsum("g, g, rg -> rg", gamma_XD, kerh.fgg, grdh.rho_1)
                        + 4 * np.einsum("rg, g -> rg", rho_X_1, kerh.fg)
                    )
                    ax_ao[idx] += (
                        + np.einsum("rg, rgu, gv -> uv", tmp_M, grdh.ao[:4], grdh.ao_0)
                    )
            ax_ao += ax_ao.swapaxes(-1, -2)
            r = C[:, si].T @ ax_ao @ C[:, sj]

            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.append(r.shape[-2])
                shape1.append(r.shape[-1])
                r.shape = shape1
            return r

        return fx


if __name__ == "__main__":

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

    nmo = nao = mol.nao
    natm = mol.natm
    nocc = mol.nelec[0]
    nvir = nmo - nocc
    so = slice(0, nocc)
    sv = slice(nocc, nmo)
    sa = slice(0, nmo)

    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.becke_scheme = dft.gen_grid.stratmann
    grids.build()

    ggah = GGAHelper(mol, "b3lypg", grids)

    X = np.random.random((3, nmo, nocc))
    print(np.allclose(
        hessian.rhf.gen_vind(ggah.scf_eng, ggah.C, ggah.mo_occ)(X),
        ggah.Ax0_Core(sa, so, sa, so)(X)
    ))
