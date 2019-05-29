import numpy as np


class DerivArray(np.ndarray):
    """
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """

    def __new__(cls, input_array, pd_func, pdU_func, pdA_only=False):
        obj = np.asarray(input_array).view(cls)
        obj.pd_func = pd_func
        obj.pdU_func = pdU_func
        obj.pdA_only = pdA_only
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._pd = NotImplemented  # type: np.ndarray
        self._pdU = NotImplemented  # type: np.ndarray
        self._pdA = NotImplemented  # type: np.ndarray
        self.pd_func = getattr(obj, "pd_func", NotImplemented)
        self.pdU_func = getattr(obj, "pdU_func", NotImplemented)
        self.pdU_only = getattr(obj, "pdA_only", NotImplemented)

    @property
    def pd(self):
        if self._pd is not NotImplemented:
            return self._pd
        else:
            self._pd = self.pd_func()
            return self._pd

    @property
    def pdU(self):
        if self._pdU is not NotImplemented:
            return self._pdU
        else:
            self._pdU = self.pdU_func()
            return self._pdU

    @property
    def pdA(self):
        if self.pdA_only:
            if self._pdA is not NotImplemented:
                return self._pdA
            else:
                self._pdA = self.pd_func() + self.pdU_func()
                return self._pdA
        else:
            return self.pd + self.pdU


class Test_DerivArray:

    def test_taged_array(self):

        class A:
            def __init__(self):
                self._a = DerivArray(np.array([0, 1]), self._get_a_pd, self._get_a_pdU)

            def _get_a_pd(self):
                return 3 + self.a

            def _get_a_pdU(self):
                return 4 + self.a

            @property
            def a(self):
                return self._a

        a = A()
        assert(np.allclose(a.a.pdA, [7, 9]))
