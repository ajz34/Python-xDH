import os

MAXCORE = "12"
MAXMEM = "10"
os.environ["MAXMEM"] = MAXMEM
os.environ["OMP_NUM_THREADS"] = MAXCORE
os.environ["OPENBLAS_NUM_THREADS"] = MAXCORE
os.environ["MKL_NUM_THREADS"] = MAXCORE
os.environ["VECLIB_MAXIMUM_THREADS"] = MAXCORE
os.environ["NUMEXPR_NUM_THREADS"] = MAXCORE


from pyxdh.utilities import NumericDiff, val_from_fchk
from pyxdh.hessian import GGAHelper, NCGGAEngine
from pyscf import scf, gto, dft, lib
import pyscf.scf.cphf
import pickle
import numpy as np
from functools import partial

np.set_printoptions(8, linewidth=1000, suppress=True)
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * 2 / 8])
np.einsum_path = partial(np.einsum_path, optimize=["greedy", 1024 ** 3 * 2 / 8])


def mol_to_grids(mol):
    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (75, 302)
    grids.becke_scheme = dft.gen_grid.stratmann
    grids.prune = None
    grids.build()
    return grids


def mol_to_xyg3_grad(mol):

    print("In mol_to_xyg3_grad", flush=True)
    nmo = mol.nao
    nocc = mol.nelec[0]
    nvir = nmo - nocc
    so = slice(0, nocc)
    sv = slice(nocc, nmo)
    sa = slice(0, nmo)

    scfh = GGAHelper(mol, "b3lypg", mol_to_grids(mol))
    nch = GGAHelper(mol, "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", mol_to_grids(mol), init_scf=False)
    ncgga = NCGGAEngine(scfh, nch)

    e, eo, ev = scfh.e, scfh.eo, scfh.ev
    C, Co, Cv = scfh.C, scfh.Co, scfh.Cv
    eri0_mo = scfh.eri0_mo
    eri1_ao = scfh.eri1_ao

    S_1_mo = scfh.S_1_mo
    F_1_mo = scfh.F_1_mo
    Ax0_Core = scfh.Ax0_Core

    D_iajb = lib.direct_sum("i - a + j - b", scfh.eo, scfh.ev, scfh.eo, scfh.ev)
    t_iajb = eri0_mo[so, sv, so, sv] / D_iajb
    T_iajb = 2 * t_iajb - t_iajb.swapaxes(1, 3)

    D_r = np.zeros((nmo, nmo))
    D_r[so, so] += - 2 * np.einsum("iakb, jakb -> ij", T_iajb, t_iajb)
    D_r[sv, sv] += 2 * np.einsum("iajc, ibjc -> ab", T_iajb, t_iajb)

    L = np.zeros((nvir, nocc))
    L += Ax0_Core(sv, so, sa, sa)(D_r)
    L -= 4 * np.einsum("jakb, ijbk -> ai", T_iajb, eri0_mo[so, so, sv, so])
    L += 4 * np.einsum("ibjc, abjc -> ai", T_iajb, eri0_mo[sv, sv, so, sv])

    D_r[sv, so] = scf.cphf.solve(Ax0_Core(sv, so, sv, so), e, scfh.mo_occ, L, max_cycle=100, tol=1e-13)[0]

    # W[I] - Correct with s1-im1 term in PySCF
    D_WI = np.zeros((nmo, nmo))
    D_WI[so, so] = - 2 * np.einsum("iakb, jakb -> ij", T_iajb, eri0_mo[so, sv, so, sv])
    D_WI[sv, sv] = - 2 * np.einsum("iajc, ibjc -> ab", T_iajb, eri0_mo[so, sv, so, sv])
    D_WI[sv, so] = - 4 * np.einsum("jakb, ijbk -> ai", T_iajb, eri0_mo[so, so, sv, so])

    # W[II] - Correct with s1-zeta term in PySCF
    # Note that zeta in PySCF includes HF energy weighted density rdm1e
    # The need of scaler 1 in D_WII[sv, so] is that Aikens use doubled P
    D_WII = np.zeros((nmo, nmo))
    D_WII[so, so] = - 0.5 * D_r[so, so] * lib.direct_sum("i + j -> ij", eo, eo)
    D_WII[sv, sv] = - 0.5 * D_r[sv, sv] * lib.direct_sum("a + b -> ab", ev, ev)
    D_WII[sv, so] = - D_r[sv, so] * eo

    # W[III] - Correct with s1-vhf_s1occ term in PySCF
    D_WIII = np.zeros((nmo, nmo))
    D_WIII[so, so] = - 0.5 * Ax0_Core(so, so, sa, sa)(D_r)

    # Summation
    D_W = D_WI + D_WII + D_WIII

    # Non-seperatable - Correct with `de` generated in PySCF code part --2e AO integrals dot 2pdm--
    D_pdm2_NS = 2 * np.einsum("iajb, ui, va, kj, lb -> uvkl", T_iajb, Co, Cv, Co, Cv)

    grad_B3LYP_MP2 = (
            + (D_r * F_1_mo).sum(axis=(-1, -2))
            + (D_W * S_1_mo).sum(axis=(-1, -2))
            + (D_pdm2_NS * eri1_ao).sum(axis=(-1, -2, -3, -4))
    )
    return ncgga.E_1 + 0.3211 * grad_B3LYP_MP2


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

    natm = mol.natm

    # First check XYG3 derivative is correct
    grad_xyg3_ref = val_from_fchk("Cartesian Gradient", "../mp2_grad/xyg3_grad.fchk").reshape((natm, 3))
    grad_xyg3_anal = mol_to_xyg3_grad(mol)
    print("XYG3 gradient, PySCF v.s. Gaussian: ", np.allclose(grad_xyg3_anal, grad_xyg3_ref, atol=1e-6, rtol=1e-4))
    print("Deviation Maximum: ", abs((grad_xyg3_anal - grad_xyg3_ref).max()))
    print("Deviation Minimum: ", abs((grad_xyg3_anal - grad_xyg3_ref).min()))

    # Then calculate XYG3 numerical hessian like that in mp2_hess_num.py
    grad_xyg3_diff = NumericDiff(mol, mol_to_xyg3_grad, deriv=2).get_numdif()
    d = {
        "grad_xyg3_diff": grad_xyg3_diff,
    }
    with open("xyg3_hess_num.dat", "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    print(grad_xyg3_diff)


if __name__ == '__main__':
    main()
