from pyscf import gto, scf, grad, lib, hessian
import pyscf.scf.cphf
from pyscf.scf import _vhf
import numpy as np
from functools import partial
import os, warnings
from utilities import timing


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
        self.C = None
        self.nmo = None
        self.nao = None
        self.nocc = None
        self.nvir = None
        self.mo_occ = None
        self.sa = None
        self.so = None
        self.sv = None
        self.e = None
        self.eo = None
        self.ev = None
        self.Co = None
        self.Cv = None
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

    # Utility functions

    def initialization_pyscf(self):
        self.scf_eng = scf.RHF(self.mol)
        self.scf_eng.conv_tol = 1e-11
        self.scf_eng.conv_tol_grad = 1e-9
        if self.init_scf:
            self.eng = self.scf_eng.kernel()
        self.scf_grad = grad.RHF(self.scf_eng)
        self.scf_hess = hessian.RHF(self.scf_eng)
        return

    def initialization_scf(self):
        self.C = self.scf_eng.mo_coeff
        self.nmo = self.C.shape[1]
        self.nao = self.C.shape[0]
        self.nocc = self.mol.nelec[0]
        self.nvir = self.nmo - self.nocc
        self.mo_occ = self.scf_eng.mo_occ
        self.sa = slice(0, self.nmo)
        self.so = slice(0, self.nocc)
        self.sv = slice(self.nocc, self.nmo)
        self.e = self.scf_eng.mo_energy
        self.eo = self.e[self.so]
        self.ev = self.e[self.sv]
        self.Co = self.C[:, self.so]
        self.Cv = self.C[:, self.sv]
        self.D = self.scf_eng.make_rdm1()
        # self.eri0_ao = self.mol.intor("int2e")
        # self.eri0_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri0_ao, self.C, self.C, self.C, self.C)
        return

    @property
    def H_0_ao(self):
        if self._H_0_ao is None:
            self._H_0_ao = self.scf_eng.get_hcore()
        return self._H_0_ao

    @property
    def H_0_mo(self):
        if self._H_0_mo is None:
            self._H_0_mo = self.C.T @ self.H_0_ao @ self.C
        return self._H_0_mo

    @property
    def F_0_ao(self):
        if self._F_0_ao is None:
            self._F_0_ao = self.scf_eng.get_fock(dm=self.D)
        return self._F_0_ao

    @property
    def F_0_mo(self):
        if self._F_0_mo is None:
            self._F_0_mo = self.C.T @ self.F_0_ao @ self.C
        return self._F_0_mo

    @property
    @timing
    def eri0_ao(self):
        warnings.warn("eri0_ao: ERI should not be stored in memory! Consider J/K engines!")
        if self._eri0_ao is None:
            self._eri0_ao = self.mol.intor("int2e")
        return self._eri0_ao

    @property
    @timing
    def eri0_mo(self):
        warnings.warn("eri0_mo: ERI AO -> MO is quite expensive!")
        if self._eri0_mo is None:
            self._eri0_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri0_ao, self.C, self.C, self.C, self.C)
        return self._eri0_mo

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

    @timing
    def Ax0_Core(self, si, sj, sk, sl, reshape=True):
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

        Returns
        -------
        fx : function which pass matrix, then return Ax @ X.
        """
        C = self.C
        sij_none = si is None and sj is None
        skl_none = sk is None and sl is None
        
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
                - 1 * self.scf_eng.get_k(dm=dm1)
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

    @timing
    def Ax1_Core(self, si, sj, sk, sl):
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
                        ax[:, sA, :] += 1 * k_1[:, sA, :]
                        ax[:, :, sA] += 1 * k_1[:, sA, :].swapaxes(-1, -2)
                        # (kt l | u v), (kt u | l v)
                        j_1A, k_1A = _vhf.direct_mapdm(
                            mol._add_suffix('int2e_ip1'), "s2kl",
                            ("ji->s1kl", "li->s1kj"),
                            dm1Bs[:, p0:p1], 3,
                            mol._atm, mol._bas, mol._env,
                            shls_slice=((shl0, shl1) + (0, mol.nbas) * 3)
                        )
                        ax -= 4 * j_1A
                        ax += k_1A + k_1A.swapaxes(-1, -2)

                        if sij_none:
                            ax_final[A, B, :, s] = ax
                        else:
                            ax_final[A, B, :, s] = C[:, si].T @ ax @ C[:, sj]

            return ax_final

        return fx

    # Values

    def get_grad(self):
        self.grad = self.scf_grad.kernel()
        return self.grad

    def get_hess(self):
        self.hess = self.scf_hess.kernel()
        return self.hess

    @property
    def H_1_ao(self):
        if self._H_1_ao is None:
            self._H_1_ao = np.array([self.scf_grad.hcore_generator()(A) for A in range(self.natm)])
        return self._H_1_ao

    @property
    def H_1_mo(self):
        if self._H_1_mo is None:
            self._H_1_mo = np.einsum("Atuv, up, vq -> Atpq", self.H_1_ao, self.C, self.C)
        return self._H_1_mo

    @property
    def F_1_ao(self):
        if self._F_1_ao is None:
            self._F_1_ao = self.scf_hess.make_h1(self.C, self.mo_occ)
        return self._F_1_ao

    @property
    def F_1_mo(self):
        if self._F_1_mo is None:
            self._F_1_mo = np.einsum("Atuv, up, vq -> Atpq", self.F_1_ao, self.C, self.C)
        return self._F_1_mo

    @property
    def S_1_ao(self):
        if self._S_1_ao is None:
            int1e_ipovlp = self.mol.intor("int1e_ipovlp")

            def get_S_S_ao(A):
                ao_matrix = np.zeros((3, self.nao, self.nao))
                sA = self.mol_slice(A)
                ao_matrix[:, sA] = -int1e_ipovlp[:, sA]
                return ao_matrix + ao_matrix.swapaxes(1, 2)

            self._S_1_ao = np.array([get_S_S_ao(A) for A in range(self.natm)])
        return self._S_1_ao

    @property
    def S_1_mo(self):
        if self._S_1_mo is None:
            self._S_1_mo = np.einsum("Atuv, up, vq -> Atpq", self.S_1_ao, self.C, self.C)
        return self._S_1_mo

    @property
    @timing
    def eri1_ao(self):
        warnings.warn("eri1_ao: 4-idx tensor ERI should be not used!", FutureWarning)
        if self._eri1_ao is None:
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
            self._eri1_ao = eri1_ao
        return self._eri1_ao

    @property
    @timing
    def eri1_mo(self):
        warnings.warn("eri1_mo: 4-idx tensor ERI should be not used!", FutureWarning)
        if self._eri1_mo is None:
            self._eri1_mo = np.einsum("Atuvkl, up, vq, kr, ls -> Atpqrs", self.eri1_ao, self.C, self.C, self.C, self.C)
        return self._eri1_mo

    @property
    def H_2_ao(self):
        if self._H_2_ao is None:
            self._H_2_ao = np.array(
                [[self.scf_hess.hcore_generator()(A, B) for B in range(self.natm)] for A in range(self.natm)])
        return self._H_2_ao

    @property
    def H_2_mo(self):
        if self._H_2_mo is None:
            self._H_2_mo = np.einsum("ABtsuv, up, vq -> ABtspq", self.H_2_ao, self.C, self.C)
        return self._H_2_mo

    @property
    def S_2_ao(self):
        if self._S_2_ao is None:
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

            self._S_2_ao = np.array([[get_S_SS_ao(A, B) for B in range(self.natm)] for A in range(self.natm)])
        return self._S_2_ao

    @property
    def S_2_mo(self):
        if self._S_2_mo is None:
            self._S_2_mo = np.einsum("ABtsuv, up, vq -> ABtspq", self.S_2_ao, self.C, self.C)
        return self._S_2_mo

    @property
    @timing
    def eri2_ao(self):
        warnings.warn("eri2_ao: 4-idx tensor ERI should be not used!", FutureWarning)
        if self._eri2_ao is None:
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

                return eri2.reshape(3, 3, nao, nao, nao, nao)

            eri2_ao = [[get_eri2(A, B) for B in range(natm)] for A in range(natm)]
            self._eri2_ao = eri2_ao
        return self._eri2_ao

    @property
    @timing
    def eri2_mo(self):
        warnings.warn("eri2_mo: 4-idx tensor ERI should be not used!", FutureWarning)
        if self._eri2_mo is None:
            self._eri2_mo = np.einsum("ABTSuvkl, up, vq, kr, ls -> ABTSpqrs", self.eri2_ao, self.C, self.C, self.C, self.C)
        return self._eri2_mo

    def get_F_2_ao_byeri2(self):
        F_2_ao = (
                self.H_2_ao
                + np.einsum("ABtsuvkl, kl -> ABtsuv", self.eri2_ao, self.D)
                - 0.5 * np.einsum("ABtsukvl, kl -> ABtsuv", self.eri2_ao, self.D)
        )
        return F_2_ao

    def get_F_2_ao_JKcontrib_(self, cx=1):

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

        JKcontrib = np.zeros((natm, natm, 3, 3, nao, nao))
        kprefix = - cx * 0.5
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
            JKcontrib[A, A, :, :, sA, :] += j_1[:, :, sA, :]
            JKcontrib[A, A] += j_1A
            JKcontrib[A, A, :, :, sA] += kprefix * k_1[:, :, sA]
            JKcontrib[A, A] += kprefix * k_1A[A]

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
                JKcontrib[A, B, :, :, sA, sB] += j_2[:, :, sA, sB]
                JKcontrib[A, B] += j_2AB
                JKcontrib[A, B, :, :, sA] += 2 * j_3A[B][:, :, sA]
                JKcontrib[B, A, :, :, sB] += 2 * j_3A[A][:, :, sB]
                JKcontrib[A, B, :, :, sA] += kprefix * k_2A[B][:, :, sA]
                JKcontrib[B, A, :, :, sB] += kprefix * k_2A[A][:, :, sB]
                JKcontrib[A, B, :, :, sA] += kprefix * k_3A[B][:, :, sA]
                JKcontrib[B, A, :, :, sB] += kprefix * k_3A[A][:, :, sB]
                JKcontrib[A, B, :, :, sA, sB] += kprefix * k_3[:, :, sB, sA].swapaxes(-1, -2)
                JKcontrib[A, B] += kprefix * k_3AB

            # A == B finalize

            JKcontrib[A, A] /= 2

        # Symmetry Finalize
        JKcontrib += JKcontrib.transpose((0, 1, 2, 3, 5, 4))
        JKcontrib += JKcontrib.transpose((1, 0, 3, 2, 4, 5))

        return JKcontrib

    @property
    def F_2_ao(self):
        if self._F_2_ao is None:
            self._F_2_ao = self.H_2_ao + self.get_F_2_ao_JKcontrib_()
        return self._F_2_ao

    @property
    def F_2_mo(self):
        if self._F_2_mo is None:
            self._F_2_mo = np.einsum("ABtsuv, up, vq -> pq", self.F_2_ao, self.C, self.C)
        return self._F_2_mo

    @property
    def B_1(self):
        if self._B_1 is None:
            sa = self.sa
            so = self.so

            self._B_1 = (
                    self.F_1_mo
                    - self.S_1_mo * self.e
                    - 0.5 * self.Ax0_Core(sa, sa, so, so)(self.S_1_mo[:, :, so, so])
            )
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
        warnings.warn("U_1: Generating total U matrix should be considered as numerical unstable!", FutureWarning)
        if self._U_1 is None:
            self._get_U_1(total_u=True)
        return self._U_1

    @timing
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
            max_cycle=500,
            tol=1e-40,
            hermi=False
        )[0]
        U_1_ai.shape = (self.natm, 3, self.nvir, self.nocc)

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

    @property
    def Xi_2(self):
        if self._Xi_2 is None:
            U_1 = self.U_1
            S_1_mo = self.S_1_mo

            self._Xi_2 = (
                    self.S_2_mo
                    + np.einsum("Atpm, Bsqm -> ABtspq", U_1, U_1)
                    + np.einsum("Bspm, Atqm -> ABtspq", U_1, U_1)
                    - np.einsum("Atpm, Bsqm -> ABtspq", S_1_mo, S_1_mo)
                    - np.einsum("Bspm, Atqm -> ABtspq", S_1_mo, S_1_mo)
            )
        return self._Xi_2

    @property
    def B_2_vo(self):
        if self._B_2_vo is None:
            sv = self.sv
            so = self.so
            sa = self.sa
            eo = self.eo
            e = self.e
            U_1 = self.U_1
            F_1_mo = self.F_1_mo
            Ax0_Core = self.Ax0_Core
            Ax1_Core = self.Ax1_Core
            self._B_2_vo = (
                # line 1
                np.einsum("ABtsuv, ua, vi -> ABtsai", self.F_2_ao, self.Cv, self.Co)
                - np.einsum("ABtsai, i -> ABtsai", self.Xi_2[:, :, :, :, sv, so], eo)
                - 0.5 * Ax0_Core(sv, so, so, so)(self.Xi_2[:, :, :, :, so, so])
                # line 2
                + np.einsum("Atpa, Bspi -> ABtsai", U_1[:, :, :, sv], F_1_mo[:, :, :, so])
                + np.einsum("Bspa, Atpi -> ABtsai", U_1[:, :, :, sv], F_1_mo[:, :, :, so])
                + np.einsum("Atpi, Bspa -> ABtsai", U_1[:, :, :, so], F_1_mo[:, :, :, sv])
                + np.einsum("Bspi, Atpa -> ABtsai", U_1[:, :, :, so], F_1_mo[:, :, :, sv])
                # line 3
                + np.einsum("Atpa, Bspi, p -> ABtsai", U_1[:, :, :, sv], U_1[:, :, :, so], e)
                + np.einsum("Bspa, Atpi, p -> ABtsai", U_1[:, :, :, sv], U_1[:, :, :, so], e)
                # line 4
                + Ax0_Core(sv, so, sa, sa)(np.einsum("Atkm, Bslm -> ABtskl", U_1[:, :, :, so], U_1[:, :, :, so]))
                # line 5
                + np.einsum("Atpa, Bspi -> ABtsai", U_1[:, :, :, sv], Ax0_Core(sa, so, sa, so)(U_1[:, :, :, so]))
                + np.einsum("Bspa, Atpi -> ABtsai", U_1[:, :, :, sv], Ax0_Core(sa, so, sa, so)(U_1[:, :, :, so]))
                # line 6
                + np.einsum("Atpi, Bspa -> ABtsai", U_1[:, :, :, so], Ax0_Core(sa, sv, sa, so)(U_1[:, :, :, so]))
                + np.einsum("Bspi, Atpa -> ABtsai", U_1[:, :, :, so], Ax0_Core(sa, sv, sa, so)(U_1[:, :, :, so]))
                # line 7
                + Ax1_Core(sv, so, sa, so)(U_1[:, :, :, so])
                + Ax1_Core(sv, so, sa, so)(U_1[:, :, :, so]).transpose((1, 0, 3, 2, 4, 5))
            )
        return self._B_2_vo

    @property
    def U_2_vo(self):
        if self._U_2_vo is None:
            B_2_vo = self.B_2_vo
            Ax0_Core = self.Ax0_Core
            sv = self.sv
            so = self.so

            # Generate v-o block of U
            U_2_vo = scf.cphf.solve(
                self.Ax0_Core(sv, so, sv, so),
                self.e,
                self.scf_eng.mo_occ,
                B_2_vo.reshape(-1, self.nvir, self.nocc),
                max_cycle=500,
                tol=1e-15,
                hermi=False
            )[0]
            U_2_vo.shape = (self.natm, self.natm, 3, 3, self.nvir, self.nocc)
            self._U_2_vo = U_2_vo
        return self._U_2_vo
