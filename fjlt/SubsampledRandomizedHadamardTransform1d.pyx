"""
Fast Johnson-Lindenstrauss Transform (FJLT)

http://people.inf.ethz.ch/kgabriel/software.html
"""

from __future__ import division
cimport numpy as np
import cython
import numpy as np
from random_projection_fast import inverse_fast_unitary_transform_fast_1d, fast_unitary_transform_fast_1d
# from random_projection_fast cimport inverse_fast_unitary_transform_fast_1d, fast_unitary_transform_fast_1d, fftw_import_wisdom_from_string, fftw_export_wisdom_to_string
import os

cdef extern from "fftw3.h":
    char *fftw_export_wisdom_to_string()
    int fftw_import_wisdom_from_string(const char *input_string)

cdef class SubsampledRandomizedHadamardTransform1d:

    def __cinit__(self, np.int_t k, bytes wisdom):
        self.k = k
        fftw_import_wisdom_from_string(wisdom)

    cdef fit(self, double[:] X):
        self.n = X.shape[0]
        self.D = np.sign(np.random.randn(self.n))
        self.srht_const = np.sqrt(self.n / self.k)
        self.S = np.random.choice(self.n, self.k, replace=False)

    cdef np.ndarray[double, ndim=1] transform(self, double[:] x):
        cdef np.ndarray[double, ndim=1] a = np.asarray(fast_unitary_transform_fast_1d(x, D=self.D))
        return self.srht_const * a[np.asarray(self.S)]

    cdef np.ndarray[double, ndim=1] inverse_transform(self, np.ndarray[double, ndim=1] a):
        cdef np.ndarray[double, ndim=1] x = np.zeros(self.n)
        x[np.asarray(self.S)] = a / self.srht_const
        return np.asarray(inverse_fast_unitary_transform_fast_1d(x, D=self.D))

    cdef np.ndarray[double, ndim=1] fit_transform(self, double[:] X):
        self.fit(X)
        return self.transform(X)

    cdef bytes get_wisdom(self):
        return fftw_export_wisdom_to_string()


def get_include():
    return os.path.dirname(os.path.realpath(__file__))

