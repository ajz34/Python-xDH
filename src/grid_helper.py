from pyscf import dft
import pyscf.dft.numint
import numpy as np
from functools import partial
import os

MAXMEM = int(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class GridHelper:

    def __init__(self, mol, grids, D):

        # Initialization Parameters
        self.mol = mol  # type: pyscf.gto.Mole
        self.grids = grids  # type: pyscf.dft.gen_grid.Grids
        self.D = D  # type: np.ndarray

        # Calculation
        nao = mol.nao
        ni = dft.numint.NumInt()
        ngrid = grids.weights.size
        grid_weight = grids.weights
        grid_ao = np.empty((20, ngrid, nao))  # 20 at first dimension is related to 3rd derivative of orbital
        current_grid_count = 0
        for ao, _, _, _ in ni.block_loop(mol, grids, nao, 3, 2000):
            grid_ao[:, current_grid_count:current_grid_count+ao.shape[1]] = ao
            current_grid_count += ao.shape[1]
        current_grid_count = None
        grid_ao_0 = grid_ao[0]
        grid_ao_1 = grid_ao[1:4]
        grid_ao_2T = grid_ao[4:10]
        XX, XY, XZ, YY, YZ, ZZ = range(4, 10)
        XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ = range(10, 20)
        grid_ao_2 = np.array([
            [grid_ao[XX], grid_ao[XY], grid_ao[XZ]],
            [grid_ao[XY], grid_ao[YY], grid_ao[YZ]],
            [grid_ao[XZ], grid_ao[YZ], grid_ao[ZZ]],
        ])
        grid_ao_3T = np.array([
            [grid_ao[XXX], grid_ao[XXY], grid_ao[XXZ], grid_ao[XYY], grid_ao[XYZ], grid_ao[XZZ]],
            [grid_ao[XXY], grid_ao[XYY], grid_ao[XYZ], grid_ao[YYY], grid_ao[YYZ], grid_ao[YZZ]],
            [grid_ao[XXZ], grid_ao[XYZ], grid_ao[XZZ], grid_ao[YYZ], grid_ao[YZZ], grid_ao[ZZZ]],
        ])
        grid_rho_01 = np.einsum("uv, rgu, gv -> rg", D, grid_ao[0:4], grid_ao_0)
        grid_rho_01[1:] *= 2
        grid_rho_0 = grid_rho_01[0]
        grid_rho_1 = grid_rho_01[1:4]
        grid_rho_2 = (
            + 2 * np.einsum("uv, rgu, wgv -> rwuv", D, grid_ao_1, grid_ao_1)
            + 2 * np.einsum("uv, rwgu, gv -> rwuv", D, grid_ao_2, grid_ao_0)
        )

        natm = mol.natm
        grid_A_rho_1 = np.zeros((natm, 3, ngrid))
        grid_A_rho_2 = np.zeros((natm, 3, 3, ngrid))
        for A in range(natm):
            _, _, p0, p1 = mol.aoslice_by_atom()[A]
            sA = slice(p0, p1)
            grid_A_rho_1[A] = - 2 * np.einsum("tgk , gl, kl -> tg ", grid_ao_1[:, :, sA], grid_ao_0, D[sA])
            grid_A_rho_2[A] = - 2 * np.einsum("trgk, gl, kl -> trg", grid_ao_2[:, :, :, sA], grid_ao_0, D[sA])
            grid_A_rho_2[A] += - 2 * np.einsum("tgk, rgl, kl -> trg", grid_ao_1[:, :, sA], grid_ao_1, D[sA])

        # Variable definition
        self.ni = ni
        self.ngrid = ngrid
        self.weight = grid_weight
        self.ao = grid_ao
        self.ao_0 = grid_ao_0
        self.ao_1 = grid_ao_1
        self.ao_2T = grid_ao_2T
        self.ao_2 = grid_ao_2
        self.ao_3T = grid_ao_3T
        self.rho_01 = grid_rho_01
        self.rho_0 = grid_rho_0
        self.rho_1 = grid_rho_1
        self.rho_2 = grid_rho_2
        self.A_rho_1 = grid_A_rho_1
        self.A_rho_2 = grid_A_rho_2
        return


class KernelHelper:

    def __init__(self, gh, xc):

        # Initialization Parameters
        self.gh = gh  # type: GridHelper
        self.xc = xc  # type: str

        # Calculation
        cx = gh.ni.hybrid_coeff(xc)
        grid_exc, grid_vxc, grid_fxc = gh.ni.eval_xc(xc, gh.rho_01, deriv=2)[:3]
        grid_fr, grid_fg = grid_vxc[0:2]
        grid_frr, grid_frg, grid_fgg = grid_fxc[0:3]
        grid_exc *= gh.weight
        grid_fr = grid_vxc[0] * gh.weight
        grid_fg = grid_vxc[1] * gh.weight
        grid_frr = grid_fxc[0] * gh.weight
        grid_frg = grid_fxc[1] * gh.weight
        grid_fgg = grid_fxc[2] * gh.weight

        # Variable definition
        self.exc = grid_exc
        self.fr = grid_fr
        self.fg = grid_fg
        self.frr = grid_frr
        self.frg = grid_frg
        self.fgg = grid_fgg
        self.vxc = grid_vxc
        self.fxc = grid_fxc
        return