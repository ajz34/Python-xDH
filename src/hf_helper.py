from pyscf import gto, scf, grad, lib, hessian
import pyscf.scf.cphf
from pyscf.scf import _vhf
import numpy as np
from functools import partial
import os, warnings, gc
from utilities import timing, gccollect

MAXMEM = float(os.getenv("MAXMEM", 2))
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * MAXMEM / 8])
np.set_printoptions(8, linewidth=1000, suppress=True)


class HFHelper:

    def __init__(self, mol, init_scf=True):
        # Basic Engine
        self.mol = mol  # type: gto.Mole
        self.init_scf = init_scf
        # mol.verbose = 0
        self.natm = mol.natm
        self.grad = None
        self.hess = None
        # Initialize
        self.scf_eng = None  # type: scf.rhf.RHF
        self.eng = None
        self.scf_grad = None  # type: grad.rhf.Gradients
        self.scf_hess = None  # type: hessian.rhf.Hessian
        # From SCF calculation
        self._C = None
        self.C = None
        self._mo_occ = None
        self.mo_occ = None
        self._e = None
        self.e = None
        self._D = None
        self.D = None
        self._F_0_ao = None
        self._F_0_mo = None
        self._H_0_ao = None
        self._H_0_mo = None
        # From gradient and hessian calculation
        self._H_1_ao = None
        self._H_1_mo = None
        self._S_1_ao = None
        self._S_1_mo = None
        self._F_1_ao = None
        self._F_1_mo = None
        self._H_2_ao = None
        self._H_2_mo = None
        self._S_2_ao = None
        self._S_2_mo = None
        self._F_2_ao = None
        self._F_2_ao_Jcontrib = None
        self._F_2_ao_Kcontrib = None
        self._F_2_mo = None
        self._B_1 = None
        self._U_1 = None
        self._U_1_vo = None
        self._U_1_ov = None
        self._Xi_2 = None
        self._B_2_vo = None
        self._U_2_vo = None
        # ERI
        self._eri0_ao = None
        self._eri0_mo = None
        self._eri1_ao = None
        self._eri1_mo = None
        self._eri2_ao = None
        self._eri2_mo = None

        # Initializer
        self.initialization_pyscf()
        if init_scf:
            self.initialization_scf()
        return

    # region Initializers

    @gccollect
    def initialization_pyscf(self):
        self.scf_eng = scf.RHF(self.mol)
        self.scf_eng.conv_tol = 1e-11
        self.scf_eng.conv_tol_grad = 1e-9
        self.scf_eng.max_cycle = 100
        if self.init_scf:
            self.eng = self.scf_eng.kernel()
            if not self.scf_eng.converged:
                warnings.warn("SCF not converged!")
        self.scf_grad = grad.RHF(self.scf_eng)
        self.scf_hess = hessian.RHF(self.scf_eng)
        return

    def initialization_scf(self):
        self.C = self.scf_eng.mo_coeff
        self.mo_occ = self.scf_eng.mo_occ
        self.e = self.scf_eng.mo_energy
        self.D = self.scf_eng.make_rdm1()
        return

    # endregion

    # region Properties

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        if self._C is None:
            self._C = C
        else:
            raise AttributeError("Once orbital coefficient is set, it should not be changed anymore.")

    @property
    def Co(self):
        return self.C[:, self.so]

    @property
    def Cv(self):
        return self.C[:, self.sv]

    @property
    def nmo(self):
        if self.C is None:
            raise ValueError("Molecular orbital number should be determined after SCF process!\nPrepare self.C first.")
        return self.C.shape[1]

    @property
    def nao(self):
        return self.mol.nao

    @property
    def nocc(self):
        return self.mol.nelec[0]

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def mo_occ(self):
        return self._mo_occ

    @mo_occ.setter
    def mo_occ(self, mo_occ):
        if self._mo_occ is None:
            self._mo_occ = mo_occ
        else:
            raise AttributeError("Once mo_occ is set, it should not be changed anymore.")

    @property
    def sa(self):
        return slice(0, self.nmo)

    @property
    def so(self):
        return slice(0, self.nocc)

    @property
    def sv(self):
        return slice(self.nocc, self.nmo)

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, e):
        if self._e is None:
            self._e = e
        else:
            raise AttributeError("Once orbital energy is set, it should not be changed anymore.")

    @property
    def eo(self):
        return self.e[self.so]

    @property
    def ev(self):
        return self.e[self.sv]

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, D):
        if self._D is None:
            self._D = D
        else:
            raise AttributeError("Once AO basis density is set, it should not be changed anymore.")

    @property
    def H_0_ao(self):
        if self._H_0_ao is None:
            self._H_0_ao = self._get_H_0_ao()
        return self._H_0_ao

    @property
    def H_0_mo(self):
        if self._H_0_mo is None:
            self._H_0_mo = self._get_H_0_mo()
        return self._H_0_mo

    @property
    def F_0_ao(self):
        if self._F_0_ao is None:
            self._F_0_ao = self._get_F_0_ao()
        return self._F_0_ao

    @property
    def F_0_mo(self):
        if self._F_0_mo is None:
            self._F_0_mo = self._get_F_0_mo()
        return self._F_0_mo

    @property
    def eri0_ao(self):
        warnings.warn("eri0_ao: ERI should not be stored in memory! Consider J/K engines!")
        if self._eri0_ao is None:
            self._eri0_ao = self._get_eri0_ao()
        return self._eri0_ao

    @property
    def eri0_mo(self):
        warnings.warn("eri0_mo: ERI AO -> MO is quite expensive!")
        if self._eri0_mo is None:
            self._eri0_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri0_ao, self.C, self.C, self.C, self.C)
        return self._eri0_mo

    @property
    def H_1_ao(self):
        if self._H_1_ao is None:
            self._H_1_ao = self._get_H_1_ao()
        return self._H_1_ao

    @property
    def H_1_mo(self):
        if self._H_1_mo is None:
            self._H_1_mo = self._get_H_1_mo()
        return self._H_1_mo

    @property
    def F_1_ao(self):
        if self._F_1_ao is None:
            self._F_1_ao = self._get_F_1_ao()
        return self._F_1_ao

    @property
    def F_1_mo(self):
        if self._F_1_mo is None:
            self._F_1_mo = self._get_F_1_mo()
        return self._F_1_mo

    @property
    def S_1_ao(self):
        if self._S_1_ao is None:
            self._S_1_ao = self._get_S_1_ao()
        return self._S_1_ao

    @property
    def S_1_mo(self):
        if self._S_1_mo is None:
            self._S_1_mo = self._get_S_1_mo()
        return self._S_1_mo

    @property
    def eri1_ao(self):
        warnings.warn("eri1_ao: 4-idx tensor ERI should be not used!")
        if self._eri1_ao is None:
            self._eri1_ao = self._get_eri1_ao()
        return self._eri1_ao

    @property
    def eri1_mo(self):
        warnings.warn("eri1_mo: 4-idx tensor ERI should be not used!")
        if self._eri1_mo is None:
            self._eri1_mo = self._get_eri1_mo()
        return self._eri1_mo

    @property
    def H_2_ao(self):
        if self._H_2_ao is None:
            self._H_2_ao = self._get_H_2_ao()
        return self._H_2_ao

    @property
    def H_2_mo(self):
        if self._H_2_mo is None:
            self._H_2_mo = self._get_H_2_mo()
        return self._H_2_mo

    @property
    def S_2_ao(self):
        if self._S_2_ao is None:
            self._S_2_ao = self._get_S_2_ao()
        return self._S_2_ao

    @property
    def S_2_mo(self):
        if self._S_2_mo is None:
            self._S_2_mo = self._get_S_2_mo()
        return self._S_2_mo

    @property
    def eri2_ao(self):
        warnings.warn("eri2_ao: 4-idx tensor ERI should be not used!")
        if self._eri2_ao is None:
            self._eri2_ao = self._get_eri2_ao()
        return self._eri2_ao

    @property
    def eri2_mo(self):
        warnings.warn("eri2_mo: 4-idx tensor ERI should be not used!")
        if self._eri2_mo is None:
            self._eri2_mo = self._get_eri2_mo()
        return self._eri2_mo

    @property
    def F_2_ao_Jcontrib(self):
        if self._F_2_ao_Jcontrib is None:
            self._F_2_ao_Jcontrib, self._F_2_ao_Kcontrib = self._get_F_2_ao_JKcontrib()
        return self._F_2_ao_Jcontrib

    @property
    def F_2_ao_Kcontrib(self):
        if self._F_2_ao_Kcontrib is None:
            self._F_2_ao_Jcontrib, self._F_2_ao_Kcontrib = self._get_F_2_ao_JKcontrib()
        return self._F_2_ao_Kcontrib

    @property
    def F_2_ao(self):
        if self._F_2_ao is None:
            self._F_2_ao = self._get_F_2_ao()
        return self._F_2_ao

    @property
    def F_2_mo(self):
        if self._F_2_mo is None:
            self._F_2_mo = self._get_F_2_mo()
        return self._F_2_mo

    @property
    def B_1(self):
        if self._B_1 is None:
            self._B_1 = self._get_B_1()
        return self._B_1

    @property
    def U_1_vo(self):
        if self._U_1_vo is None:
            self._get_U_1(total_u=False)
        return self._U_1_vo

    @property
    def U_1_ov(self):
        if self._U_1_ov is None:
            self._get_U_1(total_u=False)
        return self._U_1_ov

    @property
    def U_1(self):
        warnings.warn("U_1: Generating total U matrix should be considered as numerical unstable!")
        if self._U_1 is None:
            self._get_U_1(total_u=True)
        return self._U_1

    @property
    def Xi_2(self):
        if self._Xi_2 is None:
            self._Xi_2 = self._get_Xi_2()
        return self._Xi_2

    @property
    def B_2_vo(self):
        if self._B_2_vo is None:
            self._B_2_vo = self._get_B_2_vo()
        return self._B_2_vo

    @property
    def U_2_vo(self):
        if self._U_2_vo is None:
            self._U_2_vo = self._get_U_2_vo()
        return self._U_2_vo

    # endregion

    # region Utility functions

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

    def Ax0_Core(self, si, sj, sk, sl, cx=1, reshape=True):
        """

        Parameters
        ----------
        si : slice or None
        sj : slice or None
            ``si`` and ``sj`` should be all slice or all None. If chosen None, then return an AO base Ax(mo1).

        sk : slice or None
        sl : slice or None
            ``sk`` and ``sk`` should be all slice or all None. If chosen None, then `mo1` that passed in is assumed to
            be a density matrix.
        reshape : bool

        cx : float
            Exchange integral coefficient. Use in DFT calculation.

        Returns
        -------
        fx : function which pass matrix, then return Ax @ X.
        """
        C = self.C
        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None

        @timing
        @gccollect
        def fx(mo1_):
            mo1 = mo1_.copy()  # type: np.ndarray
            shape1 = list(mo1.shape)
            mo1.shape = (-1, shape1[-2], shape1[-1])
            if skl_none:
                dm1 = mo1
                if dm1.shape[-2] != self.nao or dm1.shape[-1] != self.nao:
                    raise ValueError("if `sk`, `sl` is None, we assume that mo1 passed in is an AO-based matrix!")
            else:
                dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.transpose((0, 2, 1))
            # Use PySCF higher functions to avoid explicit eri0_ao storage
            r = (
                + 2 * self.scf_eng.get_j(dm=dm1)
                - cx * self.scf_eng.get_k(dm=dm1)
            )
            if not sij_none:
                r = np.einsum("Auv, ui, vj -> Aij", r, C[:, si], C[:, sj])
            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.append(r.shape[-2])
                shape1.append(r.shape[-1])
                r.shape = shape1
            return r

        return fx

    @timing
    def Ax1_Core_use_eri1_ao(self, si, sj, sk, sl):
        C = self.C

        def fx(mo1):
            dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.swapaxes(-2, -1)
            r = (
                    + 2 * np.einsum("Atuvkl, Bskl, ui, vj -> ABtsij", self.eri1_ao, dm1, C[:, si], C[:, sj])
                    - 1 * np.einsum("Atukvl, Bskl, ui, vj -> ABtsij", self.eri1_ao, dm1, C[:, si], C[:, sj])
            )
            return r

        return fx

    def Ax1_Core(self, si, sj, sk, sl, cx=1):
        """
        Calculate Axt @ X.

        This function has been modified so that eri1_ao is not essential to be stored. for previous version of Ax1_Core,
        refer to Ax1_Core_use_eri1_ao.

        Parameters
        ----------
        si : slice or None
        sj : slice or None
            ``si`` and ``sj`` should be all slice or all None. If chosen None, then return an AO base Ax(mo1).

        sk : slice or None
        sl : slice or None
            ``sk`` and ``sk`` should be all slice or all None. If chosen None, then `mo1` that passed in is assumed to
            be a density matrix.

        cx : float
            Exchange integral coefficient. Use in DFT calculation.

        Returns
        -------
        fx : function which pass matrix, then return Axt @ X.
        """
        C = self.C
        _vhf = pyscf.scf._vhf
        natm = self.natm
        nao = self.nao
        mol = self.mol

        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None

        @timing
        @gccollect
        def fx(mo1):
            # It is assumed that mo1 is Bs** like
            if len(mo1.shape) != 4:
                raise ValueError("Passed in mo1 should be Bs** like, so length of mo1.shape should be 4.")
            if skl_none:
                dm1 = mo1
                if dm1.shape[-2] != self.nao or dm1.shape[-1] != self.nao:
                    raise ValueError("if `sk`, `sl` is None, we assume that mo1 passed in is an AO-based matrix!")
            else:
                dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.swapaxes(-1, -2)

            # create a return tensor
            if sij_none:
                ax_final = np.zeros((natm, dm1.shape[0], 3, dm1.shape[1], nao, nao))
            else:
                ax_final = np.zeros((natm, dm1.shape[0], 3, dm1.shape[1], si.stop - si.start, sj.stop - sj.start))

            # Actual calculation
            for B in range(dm1.shape[0]):
                for s in range(dm1.shape[1]):
                    dm1Bs = dm1[B, s]
                    # (ut v | k l), (ut k | v l)
                    j_1, k_1 = _vhf.direct_mapdm(
                        mol._add_suffix('int2e_ip1'), "s2kl",
                        ("lk->s1ij", "jk->s1il"),
                        dm1Bs, 3,
                        mol._atm, mol._bas, mol._env
                    )
                    for A in range(natm):
                        ax = np.zeros((3, nao, nao))
                        shl0, shl1, p0, p1 = mol.aoslice_by_atom()[A]
                        sA = slice(p0, p1)  # equivalent to mol_slice(A)
                        ax[:, sA, :] -= 2 * j_1[:, sA, :]
                        ax[:, :, sA] -= 2 * j_1[:, sA, :].swapaxes(-1, -2)
                        ax[:, sA, :] += cx * k_1[:, sA, :]
                        ax[:, :, sA] += cx * k_1[:, sA, :].swapaxes(-1, -2)
                        # (kt l | u v), (kt u | l v)
                        j_1A, k_1A = _vhf.direct_mapdm(
                            mol._add_suffix('int2e_ip1'), "s2kl",
                            ("ji->s1kl", "li->s1kj"),
                            dm1Bs[:, p0:p1], 3,
                            mol._atm, mol._bas, mol._env,
                            shls_slice=((shl0, shl1) + (0, mol.nbas) * 3)
                        )
                        ax -= 4 * j_1A
                        ax += cx * (k_1A + k_1A.swapaxes(-1, -2))

                        if sij_none:
                            ax_final[A, B, :, s] = ax
                        else:
                            ax_final[A, B, :, s] = C[:, si].T @ ax @ C[:, sj]

            return ax_final

        return fx

    def get_grad(self):
        self.grad = self.scf_grad.kernel()
        return self.grad

    def get_hess(self):
        self.hess = self.scf_hess.kernel()
        return self.hess

    # endregion

    # region Getting Functions

    def _get_H_0_ao(self):
        return self.scf_eng.get_hcore()

    def _get_H_0_mo(self):
        return self.C.T @ self.H_0_ao @ self.C

    def _get_F_0_ao(self):
        return self.scf_eng.get_fock(dm=self.D)

    def _get_F_0_mo(self):
        return self.C.T @ self.F_0_ao @ self.C

    @timing
    def _get_eri0_ao(self):
        return self.mol.intor("int2e")

    @gccollect
    def _get_H_1_ao(self):
        return np.array([self.scf_grad.hcore_generator()(A) for A in range(self.natm)])

    def _get_H_1_mo(self):
        return np.einsum("Atuv, up, vq -> Atpq", self.H_1_ao, self.C, self.C)

    @timing
    @gccollect
    def _get_F_1_ao(self):
        return self.scf_hess.make_h1(self.C, self.mo_occ)

    def _get_F_1_mo(self):
        return np.einsum("Atuv, up, vq -> Atpq", self.F_1_ao, self.C, self.C)

    def _get_S_1_ao(self):
        int1e_ipovlp = self.mol.intor("int1e_ipovlp")

        def get_S_S_ao(A):
            ao_matrix = np.zeros((3, self.nao, self.nao))
            sA = self.mol_slice(A)
            ao_matrix[:, sA] = -int1e_ipovlp[:, sA]
            return ao_matrix + ao_matrix.swapaxes(1, 2)

        S_1_ao = np.array([get_S_S_ao(A) for A in range(self.natm)])
        return S_1_ao

    def _get_S_1_mo(self):
        return np.einsum("Atuv, up, vq -> Atpq", self.S_1_ao, self.C, self.C)

    @timing
    def _get_eri1_ao(self):
        nao = self.nao
        natm = self.natm
        int2e_ip1 = self.mol.intor("int2e_ip1")
        eri1_ao = np.zeros((natm, 3, nao, nao, nao, nao))
        for A in range(natm):
            sA = self.mol_slice(A)
            eri1_ao[A, :, sA, :, :, :] -= int2e_ip1[:, sA]
            eri1_ao[A, :, :, sA, :, :] -= int2e_ip1[:, sA].transpose(0, 2, 1, 3, 4)
            eri1_ao[A, :, :, :, sA, :] -= int2e_ip1[:, sA].transpose(0, 3, 4, 1, 2)
            eri1_ao[A, :, :, :, :, sA] -= int2e_ip1[:, sA].transpose(0, 3, 4, 2, 1)
        return eri1_ao

    @timing
    def _get_eri1_mo(self):
        return np.einsum("Atuvkl, up, vq, kr, ls -> Atpqrs", self.eri1_ao, self.C, self.C, self.C, self.C)

    @timing
    @gccollect
    def _get_H_2_ao(self):
        return np.array([[self.scf_hess.hcore_generator()(A, B) for B in range(self.natm)] for A in range(self.natm)])

    def _get_H_2_mo(self):
        return np.einsum("ABtsuv, up, vq -> ABtspq", self.H_2_ao, self.C, self.C)

    @timing
    @gccollect
    def _get_S_2_ao(self):
        int1e_ipovlpip = self.mol.intor("int1e_ipovlpip")
        int1e_ipipovlp = self.mol.intor("int1e_ipipovlp")

        def get_S_SS_ao(A, B):
            ao_matrix = np.zeros((9, self.nao, self.nao))
            sA = self.mol_slice(A)
            sB = self.mol_slice(B)
            ao_matrix[:, sA, sB] = int1e_ipovlpip[:, sA, sB]
            if A == B:
                ao_matrix[:, sA] += int1e_ipipovlp[:, sA]
            return (ao_matrix + ao_matrix.swapaxes(1, 2)).reshape(3, 3, self.nao, self.nao)

        return np.array([[get_S_SS_ao(A, B) for B in range(self.natm)] for A in range(self.natm)])

    def _get_S_2_mo(self):
        return np.einsum("ABtsuv, up, vq -> ABtspq", self.S_2_ao, self.C, self.C)

    @timing
    def _get_eri2_ao(self):
        natm = self.natm
        nao = self.nao
        mol_slice = self.mol_slice

        int2e_ipip1 = self.mol.intor("int2e_ipip1")
        int2e_ipvip1 = self.mol.intor("int2e_ipvip1")
        int2e_ip1ip2 = self.mol.intor("int2e_ip1ip2")

        def get_eri2(A, B):
            sA, sB = mol_slice(A), mol_slice(B)
            eri2 = np.zeros((9, nao, nao, nao, nao))

            if A == B:
                eri2[:, sA, :, :, :] += int2e_ipip1[:, sA]
                eri2[:, :, sA, :, :] += int2e_ipip1[:, sA].transpose(0, 2, 1, 3, 4)
                eri2[:, :, :, sA, :] += int2e_ipip1[:, sA].transpose(0, 3, 4, 1, 2)
                eri2[:, :, :, :, sA] += int2e_ipip1[:, sA].transpose(0, 3, 4, 2, 1)
            eri2[:, sA, sB, :, :] += int2e_ipvip1[:, sA, sB]
            eri2[:, sB, sA, :, :] += np.einsum("Tijkl -> Tjikl", int2e_ipvip1[:, sA, sB])
            eri2[:, :, :, sA, sB] += np.einsum("Tijkl -> Tklij", int2e_ipvip1[:, sA, sB])
            eri2[:, :, :, sB, sA] += np.einsum("Tijkl -> Tklji", int2e_ipvip1[:, sA, sB])
            eri2[:, sA, :, sB, :] += int2e_ip1ip2[:, sA, :, sB]
            eri2[:, sB, :, sA, :] += np.einsum("Tijkl -> Tklij", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, sA, :, :, sB] += np.einsum("Tijkl -> Tijlk", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, sB, :, :, sA] += np.einsum("Tijkl -> Tklji", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sA, sB, :] += np.einsum("Tijkl -> Tjikl", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sB, sA, :] += np.einsum("Tijkl -> Tlkij", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sA, :, sB] += np.einsum("Tijkl -> Tjilk", int2e_ip1ip2[:, sA, :, sB])
            eri2[:, :, sB, :, sA] += np.einsum("Tijkl -> Tlkji", int2e_ip1ip2[:, sA, :, sB])

            return eri2.reshape((3, 3, nao, nao, nao, nao))

        return [[get_eri2(A, B) for B in range(natm)] for A in range(natm)]

    @timing
    def _get_eri2_mo(self):
        return np.einsum("ABTSuvkl, up, vq, kr, ls -> ABTSpqrs", self.eri2_ao, self.C, self.C, self.C, self.C)

    @timing
    def get_F_2_ao_byeri2(self):
        F_2_ao = (
                self.H_2_ao
                + np.einsum("ABtsuvkl, kl -> ABtsuv", self.eri2_ao, self.D)
                - 0.5 * np.einsum("ABtsukvl, kl -> ABtsuv", self.eri2_ao, self.D)
        )
        return F_2_ao

    @timing
    def _get_F_2_ao_JKcontrib(self):

        mol = self.mol
        natm = self.natm
        D = self.D
        nao = self.nao

        def reshape_only_first_dimension(mats_, d1=3, d2=3):
            rmats = []
            if isinstance(mats_, np.ndarray):
                mats = [mats_]
            else:
                mats = mats_
            for mat in mats:
                s = list(mat.shape)
                s[0] = d2
                s.insert(0, d1)
                mat.shape = tuple(s)

        Jcontrib = np.zeros((natm, natm, 3, 3, nao, nao))
        Kcontrib = np.zeros((natm, natm, 3, 3, nao, nao))
        hbas = (0, mol.nbas)

        # Atom insensitive contractions
        j_1 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ipip1'), "s2kl",
            ("lk->s1ij"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )
        j_2 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ipvip1'), "s2kl",
            ("lk->s1ij"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )
        k_1 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ipip1'), "s2kl",
            ("jk->s1il"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )
        k_3 = _vhf.direct_mapdm(
            mol._add_suffix('int2e_ip1ip2'), "s1",
            ("lj->s1ki"),
            D, 9,
            mol._atm, mol._bas, mol._env
        )

        reshape_only_first_dimension((j_1, j_2, k_1, k_3))

        # One atom sensitive contractions, multiple usage
        j_3A, k_1A, k_2A, k_3A = [], [], [], []
        for A in range(natm):
            shl0A, shl1A, p0A, p1A = mol.aoslice_by_atom()[A]
            sA, hA = slice(p0A, p1A), (shl0A, shl1A)

            j_3A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ip1ip2'), "s1",
                ("lk->s1ij"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hbas + hbas + hA + hbas)
            ))
            k_1A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ipip1'), "s2kl",
                ("li->s1kj"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hA + hbas * 3)
            ))
            k_2A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ipvip1'), "s2kl",
                ("jk->s1il"),
                D[sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hbas + hA + hbas + hbas)
            ))
            k_3A.append(_vhf.direct_mapdm(
                mol._add_suffix('int2e_ip1ip2'), "s1",
                ("jk->s1il"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hbas + hbas + hA + hbas)
            ))
        for jkA in j_3A, k_1A, k_2A, k_3A:
            reshape_only_first_dimension(jkA)

        for A in range(natm):
            shl0A, shl1A, p0A, p1A = mol.aoslice_by_atom()[A]
            sA, hA = slice(p0A, p1A), (shl0A, shl1A)

            # One atom sensitive contractions, One usage only
            j_1A = _vhf.direct_mapdm(
                mol._add_suffix('int2e_ipip1'), "s2kl",
                ("ji->s1kl"),
                D[:, sA], 9,
                mol._atm, mol._bas, mol._env,
                shls_slice=(hA + (0, mol.nbas) * 3)
            )
            reshape_only_first_dimension((j_1A,))

            # A-A manipulation
            Jcontrib[A, A, :, :, sA, :] += j_1[:, :, sA, :]
            Jcontrib[A, A] += j_1A
            Kcontrib[A, A, :, :, sA] += k_1[:, :, sA]
            Kcontrib[A, A] += k_1A[A]

            for B in range(A + 1):
                shl0B, shl1B, p0B, p1B = mol.aoslice_by_atom()[B]
                sB, hB = slice(p0B, p1B), (shl0B, shl1B)

                # Two atom sensitive contractions
                j_2AB = _vhf.direct_mapdm(
                    mol._add_suffix('int2e_ipvip1'), "s2kl",
                    ("ji->s1kl"),
                    D[sB, sA], 9,
                    mol._atm, mol._bas, mol._env,
                    shls_slice=(hA + hB + (0, mol.nbas) * 2)
                )
                k_3AB = _vhf.direct_mapdm(
                    mol._add_suffix('int2e_ip1ip2'), "s1",
                    ("ki->s1jl"),
                    D[sB, sA], 9,
                    mol._atm, mol._bas, mol._env,
                    shls_slice=(hA + (0, mol.nbas) + hB + (0, mol.nbas))
                )
                reshape_only_first_dimension((j_2AB, k_3AB))

                # A-B manipulation
                Jcontrib[A, B, :, :, sA, sB] += j_2[:, :, sA, sB]
                Jcontrib[A, B] += j_2AB
                Jcontrib[A, B, :, :, sA] += 2 * j_3A[B][:, :, sA]
                Jcontrib[B, A, :, :, sB] += 2 * j_3A[A][:, :, sB]
                Kcontrib[A, B, :, :, sA] += k_2A[B][:, :, sA]
                Kcontrib[B, A, :, :, sB] += k_2A[A][:, :, sB]
                Kcontrib[A, B, :, :, sA] += k_3A[B][:, :, sA]
                Kcontrib[B, A, :, :, sB] += k_3A[A][:, :, sB]
                Kcontrib[A, B, :, :, sA, sB] += k_3[:, :, sB, sA].swapaxes(-1, -2)
                Kcontrib[A, B] += k_3AB

            # A == B finalize

            Jcontrib[A, A] /= 2
            Kcontrib[A, A] /= 2
            gc.collect()

        # Symmetry Finalize
        Jcontrib += Jcontrib.transpose((0, 1, 2, 3, 5, 4))
        Jcontrib += Jcontrib.transpose((1, 0, 3, 2, 4, 5))
        Kcontrib += Kcontrib.transpose((0, 1, 2, 3, 5, 4))
        Kcontrib += Kcontrib.transpose((1, 0, 3, 2, 4, 5))

        return Jcontrib, Kcontrib

    def _get_F_2_ao(self):
        return self.H_2_ao + self.F_2_ao_Jcontrib - 0.5 * self.F_2_ao_Kcontrib

    def _get_F_2_mo(self):
        return np.einsum("ABtsuv, up, vq -> pq", self.F_2_ao, self.C, self.C)

    def _get_B_1(self):
        sa = self.sa
        so = self.so

        B_1 = (
            self.F_1_mo
            - self.S_1_mo * self.e
            - 0.5 * self.Ax0_Core(sa, sa, so, so)(self.S_1_mo[:, :, so, so])
        )
        return B_1

    @timing
    @gccollect
    def _get_U_1(self, total_u=True):
        """

        Parameters
        ----------
        total_u : bool, optional, default: True
            Since total U matrix is not always needed, this is an option that only generate v-o and o-v block of U
            matrix.

        Returns
        -------

        """
        B_1 = self.B_1
        S_1_mo = self.S_1_mo
        Ax0_Core = self.Ax0_Core
        sv = self.sv
        so = self.so

        # Generate v-o block of U
        U_1_ai = scf.cphf.solve(
            self.Ax0_Core(sv, so, sv, so),
            self.e,
            self.scf_eng.mo_occ,
            B_1[:, :, sv, so].reshape(-1, self.nvir, self.nocc),
            max_cycle=100,
            tol=1e-13,
            hermi=False
        )[0]
        U_1_ai.shape = (self.natm, 3, self.nvir, self.nocc)

        # Test whether converged
        conv = (
            + U_1_ai * lib.direct_sum("a - i", self.ev, self.eo)
            + self.Ax0_Core(sv, so, sv, so)(U_1_ai)
            + self.B_1[:, :, sv, so]
        )
        if abs(conv).max() > 1e-8:
            msg = "\nget_E_1: CP-HF not converged well!\nMaximum deviation: " + str(abs(conv).max())
            warnings.warn(msg)

        if not total_u:
            self._U_1_vo = U_1_ai
            self._U_1_ov = - S_1_mo[:, :, so, sv] - U_1_ai.swapaxes(2, 3)
            return self._U_1_vo
        else:
            # Generate total U
            D_pq = - lib.direct_sum("p - q -> pq", self.e, self.e) + 1e-300
            U_1_pq = np.zeros((self.natm, 3, self.nmo, self.nmo))
            U_1_pq[:, :, sv, so] = U_1_ai
            U_1_pq[:, :, so, sv] = - S_1_mo[:, :, so, sv] - U_1_pq[:, :, sv, so].swapaxes(2, 3)
            U_1_pq[:, :, so, so] = (Ax0_Core(so, so, sv, so)(U_1_ai) + B_1[:, :, so, so]) / D_pq[so, so]
            U_1_pq[:, :, sv, sv] = (Ax0_Core(sv, sv, sv, so)(U_1_ai) + B_1[:, :, sv, sv]) / D_pq[sv, sv]
            for p in range(self.nmo):
                U_1_pq[:, :, p, p] = - S_1_mo[:, :, p, p] / 2
            U_1_pq -= (U_1_pq + U_1_pq.swapaxes(2, 3) + S_1_mo) / 2
            U_1_pq -= (U_1_pq + U_1_pq.swapaxes(2, 3) + S_1_mo) / 2

            self._U_1_vo = U_1_pq[:, :, sv, so]
            self._U_1_ov = U_1_pq[:, :, so, sv]
            self._U_1 = U_1_pq
            return self._U_1

    @timing
    def _get_Xi_2(self):
        U_1 = self.U_1
        S_1_mo = self.S_1_mo

        Xi_2 = (
            self.S_2_mo
            + np.einsum("Atpm, Bsqm -> ABtspq", U_1, U_1)
            + np.einsum("Bspm, Atqm -> ABtspq", U_1, U_1)
            - np.einsum("Atpm, Bsqm -> ABtspq", S_1_mo, S_1_mo)
            - np.einsum("Bspm, Atqm -> ABtspq", S_1_mo, S_1_mo)
        )
        return Xi_2

    @timing
    def _get_B_2_vo(self):
        sv = self.sv
        so = self.so
        sa = self.sa
        eo = self.eo
        e = self.e
        U_1 = self.U_1
        F_1_mo = self.F_1_mo
        Ax0_Core = self.Ax0_Core
        Ax1_Core = self.Ax1_Core
        B_2_vo = (
            # line 1
            0.5 * np.einsum("ABtsuv, ua, vi -> ABtsai", self.F_2_ao, self.Cv, self.Co)
            - 0.5 * np.einsum("ABtsai, i -> ABtsai", self.Xi_2[:, :, :, :, sv, so], eo)
            - 0.25 * Ax0_Core(sv, so, so, so)(self.Xi_2[:, :, :, :, so, so])
            # line 2
            + np.einsum("Atpa, Bspi -> ABtsai", U_1[:, :, :, sv], F_1_mo[:, :, :, so])
            # + np.einsum("Bspa, Atpi -> ABtsai", U_1[:, :, :, sv], F_1_mo[:, :, :, so])
            + np.einsum("Atpi, Bspa -> ABtsai", U_1[:, :, :, so], F_1_mo[:, :, :, sv])
            # + np.einsum("Bspi, Atpa -> ABtsai", U_1[:, :, :, so], F_1_mo[:, :, :, sv])
            # line 3
            + np.einsum("Atpa, Bspi, p -> ABtsai", U_1[:, :, :, sv], U_1[:, :, :, so], e)
            # + np.einsum("Bspa, Atpi, p -> ABtsai", U_1[:, :, :, sv], U_1[:, :, :, so], e)
            # line 4
            + 0.5 * Ax0_Core(sv, so, sa, sa)(np.einsum("Atkm, Bslm -> ABtskl", U_1[:, :, :, so], U_1[:, :, :, so]))
            # line 5
            + np.einsum("Atpa, Bspi -> ABtsai", U_1[:, :, :, sv], Ax0_Core(sa, so, sa, so)(U_1[:, :, :, so]))
            # + np.einsum("Bspa, Atpi -> ABtsai", U_1[:, :, :, sv], Ax0_Core(sa, so, sa, so)(U_1[:, :, :, so]))
            # line 6
            + np.einsum("Atpi, Bspa -> ABtsai", U_1[:, :, :, so], Ax0_Core(sa, sv, sa, so)(U_1[:, :, :, so]))
            # + np.einsum("Bspi, Atpa -> ABtsai", U_1[:, :, :, so], Ax0_Core(sa, sv, sa, so)(U_1[:, :, :, so]))
            # line 7
            + Ax1_Core(sv, so, sa, so)(U_1[:, :, :, so])
            # + Ax1_Core(sv, so, sa, so)(U_1[:, :, :, so]).transpose((1, 0, 3, 2, 4, 5))
        )
        B_2_vo += B_2_vo.transpose((1, 0, 3, 2, 4, 5))
        return B_2_vo

    @timing
    def _get_U_2_vo(self):
        sv = self.sv
        so = self.so

        # Generate v-o block of U
        U_2_vo = scf.cphf.solve(
            self.Ax0_Core(sv, so, sv, so),
            self.e,
            self.scf_eng.mo_occ,
            self.B_2_vo.reshape(-1, self.nvir, self.nocc),
            max_cycle=100,
            tol=1e-11,
            hermi=False
        )[0]
        U_2_vo.shape = (self.natm, self.natm, 3, 3, self.nvir, self.nocc)
        return U_2_vo

    # endregion

    # region Reference Implementation

    def _refimp_grad_elec_by_mo(self):
        r"""
        Reference implementation: Hartree-Fock electronic energy gradient by MO orbital approach.

        Yamaguchi (p428, V.1)

        .. math::
            \frac{\partial}{\partial A_t} E_\mathrm{elec} = 2 h_{ii}^{A_t} + 2 (ii|jj)^{A_t} - (ij|ij)^{A_t} - 2 S_{ii}^{A_t} \varepsilon_i

        Returns
        -------
        grad_elec : np.ndarray

            2-idx Hartree-Fock electronic energy gradient:

            - :math:`A`: atom
            - :math:`t`: atomic coordinate componenent of :math:`A`

        See Also
        --------
        _refimp_grad_elec
        """
        so = self.so
        eo = self.eo
        H_1_mo, S_1_mo, eri1_mo = self.H_1_mo, self.S_1_mo, self.eri1_mo
        grad_elec = (
            + 2 * H_1_mo[:, :, so, so].trace(0, 2, 3)
            + 2 * eri1_mo[:, :, so, so, so, so].trace(0, 2, 3).trace(0, 2, 3)
            - eri1_mo[:, :, so, so, so, so].trace(0, 2, 4).trace(0, 2, 3)
            - 2 * (S_1_mo[:, :, so, so].diagonal(0, 2, 3) * eo).sum(axis=2)
        )
        return grad_elec

    def _refimp_grad_elec(self):
        r"""
        Reference implementation: Hartree-Fock electronic energy gradient.

        Yamaguchi (p428, V.1)

        .. math::
            \frac{\partial}{\partial A_t} E_\mathrm{elec} = h_{\mu \nu}^{A_t} D_{\mu \nu} + \frac{1}{2} (\mu \nu | \kappa \lambda)^{A_t} D_{\mu \nu} D_{\kappa \lambda} - \frac{1}{4} (\mu \kappa | \nu \lambda)^{A_t} D_{\mu \nu} D_{\kappa \lambda} - 2 S_{\mu \nu} C_{\mu i} \varepsilon_i C_{\nu i}

        Returns
        -------
        grad_elec : np.ndarray

            2-idx Hartree-Fock electronic energy gradient:

            - :math:`A`: atom
            - :math:`t`: atomic coordinate componenent of :math:`A`
        """
        H_1_ao, S_1_ao, eri1_ao = self.H_1_ao, self.S_1_ao, self.eri1_ao
        Co, eo, D = self.Co, self.eo, self.D
        grad_elec = (
            + np.einsum("Atuv, uv -> At", H_1_ao, D)
            + 0.5 * np.einsum("Atuvkl, uv, kl -> At", eri1_ao, D, D)
            - 0.25 * np.einsum("Atukvl, uv, kl -> At", eri1_ao, D, D)
            - 2 * np.einsum("Atuv, ui, i, vi -> At", S_1_ao, Co, eo, Co)
        )
        return grad_elec

    def _refimp_grad_nuc(self):
        r"""
        Reference implementation: Nucleus energy gradient.

        If variables defined

        .. math::
            Z_{MN} &= Z_M Z_N \\
            V_{MNt} &= M_t - N_t \\
            r_{MN} &= | \boldsymbol{M} - \boldsymbol{N} |

        Then gradient of nucleus energy can be expressed as

        .. math::
            \frac{\partial}{\partial A_t} E_\mathrm{nuc} = - \frac{Z_{AM}}{r_{AM}^3} V_{AMt}

        Returns
        -------
        grad_nuc : np.ndarray

            2-idx Nucleus energy gradient:

            - :math:`A`: atom
            - :math:`t`: atomic coordinate componenent of :math:`A`
        """
        mol = self.mol
        natm = self.natm
        nuc_Z = np.einsum("M, N -> MN", mol.atom_charges(), mol.atom_charges())
        nuc_V = lib.direct_sum("Mt - Nt -> MNt", mol.atom_coords(), mol.atom_coords())
        nuc_rinv = 1 / (np.linalg.norm(nuc_V, axis=2) + np.diag([np.inf] * natm))
        grad_nuc = - np.einsum("AM, AM, AMt -> At", nuc_Z, nuc_rinv ** 3, nuc_V)
        return grad_nuc

    def _refimp_grad(self):
        r"""
        Reference implementation: Hartree-Fock total energy gradient.

        Simply addition of gradient component of electronic energy and neucleus energy.

        .. math::
            \frac{\partial}{\partial A_t} E_\mathrm{total} = \frac{\partial}{\partial A_t} E_\mathrm{elec} + \frac{\partial}{\partial A_t} E_\mathrm{nuc}

        Returns
        -------
        scf_grad : np.ndarray

            2-idx Hartree-Fock total energy gradient:

            - :math:`A`: atom
            - :math:`t`: atomic coordinate componenent of :math:`A`

        See Also
        --------
        _refimp_grad_elec
        _refimp_grad_nuc
        """
        scf_grad = self._refimp_grad_elec() + self._refimp_grad_nuc()
        return scf_grad

    def _refimp_hess_elec(self):
        r"""
        Reference implementation: Hartree-Fock electronic energy hessian.

        Yamaguchi (p428, V.2)

        .. math::
            \frac{\partial^2}{\partial A_t \partial B_s} E_\mathrm{elec}
            &= h_{\mu \nu}^{A_t B_s} + \frac{1}{2} (\mu \nu | \kappa \lambda)^{A_t B_s} D_{\mu \nu} D_{\kappa \lambda} - \frac{1}{4} (\mu \kappa | \nu \lambda)^{A_t B_s} D_{\mu \nu} D_{\kappa \lambda} - 2 \xi_{ii}^{A_t B_s} \varepsilon_i \\
            &\quad + 4 U_{pi}^{B_s} F_{pi}^{A_t} + 4 U_{pi}^{A_t} F_{pi}^{B_s} + 4 U_{pi}^{A_t} U_{pi}^{B_s} \varepsilon_p + 4 U_{pi}^{A_t} A_{pi, qj} U_{qj}^{B_s}

        Returns
        -------
        hess_elec : np.ndarray

            4-idx Hartree-Fock electronic energy hessian:

            - :math:`A`: atom
            - :math:`B`: atom
            - :math:`t`: atomic coordinate componenent of :math:`A`
            - :math:`s`: atomic coordinate componenent of :math:`B`
        """
        so, sa = self.so, self.sa
        D, e, eo = self.D, self.e, self.eo
        F_1_mo = self.F_1_mo
        eri2_ao = self.eri2_ao
        H_2_ao, S_2_ao, Xi_2 = self.H_2_ao, self.S_2_ao, self.Xi_2
        U_1 = self.U_1
        Ax0_Core = self.Ax0_Core
        hess_elec = (
            + np.einsum("ABtsuv, uv -> ABts", H_2_ao, D)
            + 0.5 * np.einsum("ABtsuvkl, uv, kl -> ABts", eri2_ao, D, D)
            - 0.25 * np.einsum("ABtsukvl, uv, kl -> ABts", eri2_ao, D, D)
            - 2 * np.einsum("ABtsi, i -> ABts", Xi_2.diagonal(0, 4, 5)[:, :, :, :, so], eo)
            + 4 * np.einsum("Bspi, Atpi -> ABts", U_1[:, :, :, so], F_1_mo[:, :, :, so])
            + 4 * np.einsum("Atpi, Bspi -> ABts", U_1[:, :, :, so], F_1_mo[:, :, :, so])
            + 4 * np.einsum("Atpi, Bspi, p -> ABts", U_1[:, :, :, so], U_1[:, :, :, so], e)
            + 4 * np.einsum("Atpi, Bspi -> ABts", U_1[:, :, :, so], Ax0_Core(sa, so, sa, so)(U_1[:, :, :, so]))
        )
        return hess_elec

    def _refimp_hess_nuc(self):
        r"""
        Reference implementation: Nucleus energy hessian.

        If variables defined

        .. math::
            Z_{MN} &= Z_M Z_N \\
            V_{MNt} &= M_t - N_t \\
            r_{MN} &= | \boldsymbol{M} - \boldsymbol{N} |

        Then hessian of nucleus energy can be expressed as

        .. math::
            \frac{\partial^2 E_\mathrm{nuc}}{\partial_{A_t} \partial_{B_s}} =
            - 3 \frac{Z_{AB}}{r_{AB}^5} V_{ABt} V_{ABs}
            + 3 \frac{Z_{AM}}{r_{AM}^5} V_{AMt} V_{AMs}
            + \frac{Z_{AB}}{r_{AB}^3}
            - \frac{Z_{AM}}{r_{AM}^3}

        Returns
        -------
        hess_nuc : np.ndarray

            4-idx Nucleus energy hessian:

            - :math:`A`: atom
            - :math:`B`: atom
            - :math:`t`: atomic coordinate componenent of :math:`A`
            - :math:`s`: atomic coordinate componenent of :math:`B`

        See Also
        --------
        _refimp_grad_nuc
        """
        mol = self.mol
        natm = self.natm
        nuc_Z = np.einsum("M, N -> MN", mol.atom_charges(), mol.atom_charges())
        nuc_V = lib.direct_sum("Mt - Nt -> MNt", mol.atom_coords(), mol.atom_coords())
        nuc_rinv = 1 / (np.linalg.norm(nuc_V, axis=2) + np.diag([np.inf] * natm))
        mask_atm = np.eye(natm)[:, :, None, None]
        mask_3D = np.eye(3)[None, None, :, :]
        hess_nuc = (
            - 3 * np.einsum("AB, AB, ABt, ABs -> ABts", nuc_Z, nuc_rinv ** 5, nuc_V, nuc_V)
            + 3 * np.einsum("AM, AM, AMt, AMs -> Ats", nuc_Z, nuc_rinv ** 5, nuc_V, nuc_V) * mask_atm
            + np.einsum("AB, AB -> AB", nuc_Z, nuc_rinv ** 3)[:, :, None, None] * mask_3D
            - np.einsum("AM, AM -> A", nuc_Z, nuc_rinv ** 3)[:, None, None, None] * mask_atm * mask_3D
        )
        return hess_nuc

    def _refimp_hess(self):
        r"""
        Reference implementation: Hartree-Fock total energy hessian.

        Simply addition of hessian component of electronic energy and neucleus energy.

        .. math::
            \frac{\partial^2}{\partial A_t \partial B_s} E_\mathrm{total} = \frac{\partial^2}{\partial A_t \partial B_s} E_\mathrm{elec} + \frac{\partial^2}{\partial A_t \partial B_s} E_\mathrm{nuc}

        Returns
        -------
        scf_hess : np.ndarray

            4-idx Nucleus energy hessian:

            - :math:`A`: atom
            - :math:`B`: atom
            - :math:`t`: atomic coordinate componenent of :math:`A`
            - :math:`s`: atomic coordinate componenent of :math:`B`

            Symmetric tensor: :math:`A_t, B_s`
        """
        scf_hess = self._refimp_hess_elec() + self._refimp_hess_nuc()
        return scf_hess

    def _refimp_H_0_ao(self):
        r"""
        Reference implementation: Hamiltonian core.

        .. math::
            h_{\mu \nu} = t_{\mu \nu} + v_{\mu \nu}

        Returns
        -------
        H_0_ao : np.ndarray

            2-idx Hamiltonian core :math:`h_{\mu \nu}`.

            - :math:`\mu`: atomic orbital
            - :math:`\nu`: atomic orbital

            Symmetric tensor: :math:`\mu, \nu`
        """
        mol = self.mol
        H_0_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        return H_0_ao

    def _refimp_J_0_ao(self, X):
        r"""
        Reference implementation: Coulumb integral.

        .. math::
            J_{\mu \nu}[X_{\kappa \lambda}] = (\mu \nu | \kappa \lambda) X_{\kappa \lambda}

        Parameters
        ----------
        X : np.ndarray

            2-idx General density matrix :math:`X_{\mu \nu}`

            - :math:`\mu`: atomic orbital
            - :math:`\nu`: atomic orbital

        Returns
        -------
        J_0_ao : np.ndarray

            2-idx Coulumb integral :math:`J_{\mu \nu}[X_{\kappa \lambda}]`

            - :math:`\mu`: atomic orbital
            - :math:`\nu`: atomic orbital

            Symmetric tensor: :math:`\mu, \nu`
        """
        J_0_ao = np.einsum("uvkl, kl -> uv", self.eri0_ao, X)
        return J_0_ao

    def _refimp_K_0_ao(self, X):
        r"""
        Reference implementation: Exchange integral (with symmetric general density $X_{\kappa \lambda}$).

        .. math::
            K_{\mu \nu}[X_{\kappa \lambda}] = (\mu \kappa | \nu \lambda) X_{\kappa \lambda}

        Parameters
        ----------
        X : np.ndarray

            2-idx General density matrix :math:`X_{\mu \nu}`

            - :math:`\mu`: atomic orbital
            - :math:`\nu`: atomic orbital

            Symmetric tensor condition: :math:`\mu, \nu`

        Returns
        -------
        K_0_ao : np.ndarray

            2-idx Exchange integral :math:`K_{\mu \nu}[X_{\kappa \lambda}]`

            - :math:`\mu`: atomic orbital
            - :math:`\nu`: atomic orbital

            Symmetric tensor: :math:`\mu, \nu`
        """
        K_0_ao = np.einsum("ukvl, kl -> uv", self.eri0_ao, X)
        return K_0_ao

    def _refimp_F_0_ao(self, X):
        r"""
        Reference implementation: Fock matrix (with symmetric general density $X_{\kappa \lambda}$).

        .. math::
            F_{\mu \nu}[X_{\kappa \lambda}] = h_{\mu \nu} + J_{\mu \nu}[X_{\kappa \lambda}] - \frac{1}{2} K_{\mu \nu}[X_{\kappa \lambda}]

        Parameters
        ----------
        X : np.ndarray

            2-idx General density matrix :math:`X_{\mu \nu}`

            - :math:`\mu`: atomic orbital
            - :math:`\nu`: atomic orbital

            Symmetric tensor condition: :math:`\mu, \nu`

        Returns
        -------
        F_0_ao : np.ndarray

            2-idx Fock matrix :math:`F_{\mu \nu}[X_{\kappa \lambda}]`

            - :math:`\mu`: atomic orbital
            - :math:`\nu`: atomic orbital

            Symmetric tensor: :math:`\mu, \nu`

        See Also
        --------
        _refimp_H_0_ao
        _refimp_J_0_ao
        _refimp_K_0_ao
        """
        F_0_ao = (
            + self._refimp_H_0_ao()
            + self._refimp_J_0_ao(X)
            - 0.5 * self._refimp_K_0_ao(X)
        )
        return F_0_ao

    # endregion
