from numpy.core.multiarray import ndarray

from pyscf import gto, scf, grad, lib, hessian
import pyscf.scf.cphf
import numpy as np
from functools import partial

np.set_printoptions(8, linewidth=1000, suppress=True)
np.einsum = partial(np.einsum, optimize=["greedy", 1024 ** 3 * 2 / 8])


class HFHelper:

    def __init__(self, mol):
        # Basic Engine
        self.mol = mol  # type: gto.Mole
        mol.verbose = 0
        self.natm = mol.natm
        self.grad = None
        self.hess = None
        # Initialize
        self.scf_eng = scf.RHF(self.mol)
        self.scf_eng.conv_tol = 1e-13
        self.eng = self.scf_eng.kernel()
        self.scf_grad = grad.RHF(self.scf_eng)
        self.scf_hess = hessian.RHF(self.scf_eng)
        # From SCF calculation
        self.C = self.scf_eng.mo_coeff
        self.nmo = self.C.shape[1]
        self.nao = self.C.shape[0]
        self.nocc = mol.nelec[0]
        self.nvir = self.nmo - self.nocc
        self.sa = slice(0, self.nmo)
        self.so = slice(0, self.nocc)
        self.sv = slice(self.nocc, self.nmo)
        self.e = self.scf_eng.mo_energy
        self.eo = self.e[self.so]
        self.ev = self.e[self.sv]
        self.Co = self.C[:, self.so]
        self.Cv = self.C[:, self.sv]
        self.D = self.scf_eng.make_rdm1()
        self.F_0_ao = self.scf_eng.get_fock()
        self.F_0_mo = self.C.T @ self.F_0_ao @ self.C
        self.H_0_ao = self.scf_eng.get_hcore()
        self.H_0_mo = self.C.T @ self.H_0_ao @ self.C
        self.eri0_ao = mol.intor("int2e")
        self.eri0_mo = np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri0_ao, self.C, self.C, self.C, self.C)
        # From gradient and hessian calculation
        self.H_1_ao = None
        self.H_1_mo = None
        self.S_1_ao = None
        self.S_1_mo = None
        self.F_1_ao = None
        self.F_1_mo = None
        self.eri1_ao = None
        self.eri1_mo = None
        self.H_2_ao = None
        self.H_2_mo = None
        self.S_2_ao = None
        self.S_2_mo = None
        self.F_2_ao = None
        self.F_2_mo = None
        self.eri2_ao = None
        self.eri2_mo = None
        self.B_1 = None
        self.U_1 = None
        self.Xi_2 = None
        self.B_2_vo = None
        self.U_2_vo = None
        return

    # Utility functions

    def mol_slice(self, atm_id):
        _, _, p0, p1 = self.mol.aoslice_by_atom()[atm_id]
        return slice(p0, p1)

#    def Ax0_Core(self, si, sj, sk, sl, reshape=True):
#        def fx(mo1_):
#            mo1 = mo1_.copy()  # type: ndarray
#            shape1 = list(mo1.shape)
#            mo1.shape = (-1, shape1[-2], shape1[-1])
#            r = (
#                    4 * np.einsum("ijkl, Akl -> Aij", self.eri0_mo[si, sj, sk, sl], mo1)
#                    - np.einsum("ikjl, Akl -> Aij", self.eri0_mo[si, sk, sj, sl], mo1)
#                    - np.einsum("iljk, Akl -> Aij", self.eri0_mo[si, sl, sj, sk], mo1)
#            )
#            if reshape:
#                shape1.pop()
#                shape1.pop()
#                shape1.append(r.shape[-2])
#                shape1.append(r.shape[-1])
#                r.shape = shape1
#            return r
#
#        return fx

    def Ax0_Core(self, si, sj, sk, sl, reshape=True):
        C = self.C
        
        def fx(mo1_):
            mo1 = mo1_.copy()  # type: ndarray
            shape1 = list(mo1.shape)
            mo1.shape = (-1, shape1[-2], shape1[-1])
            dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.transpose(0, 2, 1)
            r = (
                    + 2 * np.einsum("uvkl, Akl, ui, vj -> Aij", self.eri0_ao, dm1, C[:, si], C[:, sj])
                    - 1 * np.einsum("ukvl, Akl, ui, vj -> Aij", self.eri0_ao, dm1, C[:, si], C[:, sj])
            )
            if reshape:
                shape1.pop()
                shape1.pop()
                shape1.append(r.shape[-2])
                shape1.append(r.shape[-1])
                r.shape = shape1
            return r

        return fx

#    def Ax1_Core(self, si, sj, sk, sl):
#        if self.eri1_mo is None:
#            self.get_eri1_mo()
#
#        def fx(mo1):
#            r = (
#                    4 * np.einsum("Atijkl, Bskl -> ABtsij", self.eri1_mo[:, :, si, sj, sk, sl], mo1)
#                    - np.einsum("Atikjl, Bskl -> ABtsij", self.eri1_mo[:, :, si, sk, sj, sl], mo1)
#                    - np.einsum("Atiljk, Bskl -> ABtsij", self.eri1_mo[:, :, si, sl, sj, sk], mo1)
#            )
#            return r
#
#        return fx

    def Ax1_Core(self, si, sj, sk, sl):
        C = self.C
        if self.eri1_ao is None:
            self.get_eri1_ao()

        def fx(mo1):
            dm1 = C[:, sk] @ mo1 @ C[:, sl].T
            dm1 += dm1.swapaxes(-2, -1)
            r = (
                    + 2 * np.einsum("Atuvkl, Bskl, ui, vj -> ABtsij", self.eri1_ao, dm1, C[:, si], C[:, sj])
                    - 1 * np.einsum("Atukvl, Bskl, ui, vj -> ABtsij", self.eri1_ao, dm1, C[:, si], C[:, sj])
            )
            return r

        return fx

    # Values

    def get_grad(self):
        self.grad = self.scf_grad.kernel()
        return self.grad

    def get_hess(self):
        self.hess = self.scf_hess.kernel()
        return self.hess

    def get_H_1_ao(self):
        self.H_1_ao = np.array([self.scf_grad.hcore_generator()(A) for A in range(self.natm)])
        return self.H_1_ao

    def get_H_1_mo(self):
        if self.H_1_ao is None:
            self.get_H_1_ao()
        self.H_1_mo = np.einsum("Atuv, up, vq -> Atpq", self.H_1_ao, self.C, self.C)
        return self.H_1_mo

    def get_F_1_ao(self):
        self.F_1_ao = self.scf_hess.make_h1(self.C, self.scf_eng.mo_occ)
        return self.F_1_ao

    def get_F_1_mo(self):
        if self.F_1_ao is None:
            self.get_F_1_ao()
        self.F_1_mo = np.einsum("Atuv, up, vq -> Atpq", self.F_1_ao, self.C, self.C)
        return self.F_1_mo

    def get_S_1_ao(self):
        int1e_ipovlp = self.mol.intor("int1e_ipovlp")

        def get_S_S_ao(A):
            ao_matrix = np.zeros((3, self.nao, self.nao))
            sA = self.mol_slice(A)
            ao_matrix[:, sA] = -int1e_ipovlp[:, sA]
            return ao_matrix + ao_matrix.swapaxes(1, 2)

        self.S_1_ao = np.array([get_S_S_ao(A) for A in range(self.natm)])
        return self.S_1_ao

    def get_S_1_mo(self):
        if self.S_1_ao is None:
            self.get_S_1_ao()
        self.S_1_mo = np.einsum("Atuv, up, vq -> Atpq", self.S_1_ao, self.C, self.C)
        return self.S_1_mo

    def get_eri1_ao(self):
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
        self.eri1_ao = eri1_ao
        return self.eri1_ao

    def get_eri1_mo(self):
        if self.eri1_ao is None:
            self.get_eri1_ao()
        self.eri1_mo = np.einsum("Atuvkl, up, vq, kr, ls -> Atpqrs", self.eri1_ao, self.C, self.C, self.C, self.C)
        return self.eri1_mo

    def get_H_2_ao(self):
        self.H_2_ao = np.array(
            [[self.scf_hess.hcore_generator()(A, B) for B in range(self.natm)] for A in range(self.natm)])
        return self.H_2_ao

    def get_H_2_mo(self):
        if self.H_2_ao is None:
            self.get_H_2_ao()
        self.H_2_mo = np.einsum("ABtsuv, up, vq -> ABtspq", self.H_2_ao, self.C, self.C)
        return self.H_2_mo

    def get_S_2_ao(self):
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

        self.S_2_ao = np.array([[get_S_SS_ao(A, B) for B in range(self.natm)] for A in range(self.natm)])
        return self.S_2_ao

    def get_S_2_mo(self):
        if self.S_2_ao is None:
            self.get_S_2_ao()
        self.S_2_mo = np.einsum("ABtsuv, up, vq -> ABtspq", self.S_2_ao, self.C, self.C)
        return self.S_2_mo

    def get_eri2_ao(self):
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
        self.eri2_ao = eri2_ao
        return self.eri2_ao

    def get_eri2_mo(self):
        if self.eri2_ao is None:
            self.get_eri2_ao()
        self.eri2_mo = np.einsum("ABTSuvkl, up, vq, kr, ls -> ABTSpqrs", self.eri2_ao, self.C, self.C, self.C, self.C)
        return self.eri2_mo

    def get_F_2_ao(self):
        if self.H_2_ao is None:
            self.get_H_2_ao()
        if self.eri2_ao is None:
            self.get_eri2_ao()
        self.F_2_ao = (
                self.H_2_ao
                + np.einsum("ABtsuvkl, kl -> ABtsuv", self.eri2_ao, self.D)
                - 0.5 * np.einsum("ABtsukvl, kl -> ABtsuv", self.eri2_ao, self.D)
        )
        return self.F_2_ao

    def get_F_2_mo(self):
        if self.F_2_ao is None:
            self.get_F_2_ao()
        self.F_2_mo = np.einsum("ABtsuv, up, vq -> pq", self.F_2_ao, self.C, self.C)
        return self.F_2_mo

    def get_B_1(self):
        if self.F_1_mo is None:
            self.get_F_1_mo()
        if self.S_1_mo is None:
            self.get_S_1_mo()
        sa = self.sa
        so = self.so

        self.B_1 = (
                self.F_1_mo
                - self.S_1_mo * self.e
                - 0.5 * self.Ax0_Core(sa, sa, so, so)(self.S_1_mo[:, :, so, so])
        )
        return self.B_1

    def get_U_1(self):
        if self.B_1 is None:
            self.get_B_1()
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

        # Generate total U
        D_pq = - lib.direct_sum("p - q -> pq", self.e, self.e) + 1e-300
        U_1_pq = np.zeros((self.natm, 3, self.nmo, self.nmo))
        U_1_pq[:, :, sv, so] = U_1_ai
        U_1_pq[:, :, so, sv] = - S_1_mo[:, :, so, sv] - U_1_pq[:, :, sv, so].swapaxes(2, 3)
        U_1_pq[:, :, so, so] = (Ax0_Core(so, so, sv, so)(U_1_ai) + B_1[:, :, so, so]) / D_pq[so, so]
        U_1_pq[:, :, sv, sv] = (Ax0_Core(sv, sv, sv, so)(U_1_ai) + B_1[:, :, sv, sv]) / D_pq[sv, sv]
        for p in range(self.nmo):
            U_1_pq[:, :, p, p] = - S_1_mo[:, :, p, p] / 2
        # print(np.linalg.norm(U_1_pq + U_1_pq.swapaxes(2, 3) + S_S_pq))
        U_1_pq -= (U_1_pq + U_1_pq.swapaxes(2, 3) + S_1_mo) / 2
        # print(np.linalg.norm(U_1_pq + U_1_pq.swapaxes(2, 3) + S_S_pq))
        U_1_pq -= (U_1_pq + U_1_pq.swapaxes(2, 3) + S_1_mo) / 2
        # print(np.linalg.norm(U_1_pq + U_1_pq.swapaxes(2, 3) + S_S_pq))

        self.U_1 = U_1_pq
        return self.U_1

    def get_Xi_2(self):
        if self.S_2_mo is None:
            self.get_S_2_mo()
        if self.U_1 is None:
            self.get_U_1()
        U_1 = self.U_1
        S_1_mo = self.S_1_mo

        self.Xi_2 = (
                self.S_2_mo
                + np.einsum("Atpm, Bsqm -> ABtspq", U_1, U_1)
                + np.einsum("Bspm, Atqm -> ABtspq", U_1, U_1)
                - np.einsum("Atpm, Bsqm -> ABtspq", S_1_mo, S_1_mo)
                - np.einsum("Bspm, Atqm -> ABtspq", S_1_mo, S_1_mo)
        )
        return self.Xi_2

    def get_B_2_vo(self):
        if self.F_2_ao is None:
            self.get_F_2_ao()
        if self.Xi_2 is None:
            self.get_Xi_2()
        if self.F_1_mo is None:
            self.get_F_1_mo()
        sv = self.sv
        so = self.so
        sa = self.sa
        eo = self.eo
        e = self.e
        U_1 = self.U_1
        F_1_mo = self.F_1_mo
        eri0_mo = self.eri0_mo
        Ax0_Core = self.Ax0_Core
        Ax1_Core = self.Ax1_Core
        self.B_2_vo = (
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
            + np.einsum("Atkm, Bslm, ijkl -> ABtsij", U_1[:, :, :, so], U_1[:, :, :, so], 4 * eri0_mo[sv, so, :, :])
            + np.einsum("Atkm, Bslm, ikjl -> ABtsij", U_1[:, :, :, so], U_1[:, :, :, so], - eri0_mo[sv, :, so, :])
            + np.einsum("Atkm, Bslm, iljk -> ABtsij", U_1[:, :, :, so], U_1[:, :, :, so], - eri0_mo[sv, :, so, :])
            # line 5
            + np.einsum("Atpa, Bspi -> ABtsai", U_1[:, :, :, sv], Ax0_Core(sa, so, sa, so)(U_1[:, :, :, so]))
            + np.einsum("Bspa, Atpi -> ABtsai", U_1[:, :, :, sv], Ax0_Core(sa, so, sa, so)(U_1[:, :, :, so]))
            # line 6
            + np.einsum("Atpi, Bspa -> ABtsai", U_1[:, :, :, so], Ax0_Core(sa, sv, sa, so)(U_1[:, :, :, so]))
            + np.einsum("Bspi, Atpa -> ABtsai", U_1[:, :, :, so], Ax0_Core(sa, sv, sa, so)(U_1[:, :, :, so]))
            # line 7
            + Ax1_Core(sv, so, sa, so)(U_1[:, :, :, so])
            + Ax1_Core(sv, so, sa, so)(U_1[:, :, :, so]).transpose(1, 0, 3, 2, 4, 5)
        )
        return self.B_2_vo

    def get_U_2_vo(self):
        if self.B_2_vo is None:
            self.get_B_2_vo()
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
        self.U_2_vo = U_2_vo
        return self.U_2_vo
        