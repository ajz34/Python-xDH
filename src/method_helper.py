from hf_helper import HFHelper


class MethodHelper:

    def __init__(self, hfh):
        """

        Parameters
        ----------
        hfh : HFHelper
        """
        self.natm = hfh.natm
        self.eng = hfh.eng
        # From SCF calculation
        self.C = hfh.C
        self.nmo = hfh.nmo
        self.nao = hfh.nao
        self.nocc = hfh.nocc
        self.nvir = hfh.nvir
        self.sa = hfh.sa
        self.so = hfh.so
        self.sv = hfh.sv
        self.e = hfh.e
        self.eo = hfh.eo
        self.ev = hfh.ev
        self.Co = hfh.Co
        self.Cv = hfh.Cv
        self.D = hfh.D
        self.F_0_ao = hfh.F_0_ao
        self.F_0_mo = hfh.F_0_mo
        self.H_0_ao = hfh.H_0_ao
        self.H_0_mo = hfh.H_0_mo
        self.eri0_ao = hfh.eri0_ao
        self.eri0_mo = hfh.eri0_mo
        # From gradient and hessian calculation
        self.H_1_ao = hfh.H_1_ao
        self.H_1_mo = hfh.H_1_mo
        self.S_1_ao = hfh.S_1_ao
        self.S_1_mo = hfh.S_1_mo
        self.F_1_ao = hfh.F_1_ao
        self.F_1_mo = hfh.F_1_mo
        self.eri1_ao = hfh.eri1_ao
        self.eri1_mo = hfh.eri1_mo
        self.H_2_ao = hfh.H_2_ao
        self.H_2_mo = hfh.H_2_mo
        self.S_2_ao = hfh.S_2_ao
        self.S_2_mo = hfh.S_2_mo
        self.F_2_ao = hfh.F_2_ao
        self.F_2_mo = hfh.F_2_mo
        self.eri2_ao = hfh.eri2_ao
        self.eri2_mo = hfh.eri2_mo
        self.B_1 = hfh.B_1
        self.U_1 = hfh.U_1
        self.Xi_2 = hfh.Xi_2
        self.B_2_vo = hfh.B_2_vo
        self.U_2_vo = hfh.U_2_vo
