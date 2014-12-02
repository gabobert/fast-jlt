"""
Fast Johnson-Lindenstrauss Transform (FJLT)

http://people.inf.ethz.ch/kgabriel/software.html
"""

from __future__ import division
import numpy as np
cimport numpy as np

cdef class SubsampledRandomizedFourrierTransform1d:
    cdef np.int_t k
    cdef np.int_t n
    cdef double[:] D
    cdef double srht_const
    cdef np.int_t [:] S

    cdef fit(self, double[:] X)
    cdef np.ndarray[double, ndim=1] transform(self, double[:] x)
    cdef np.ndarray[double, ndim=1] inverse_transform(self, np.ndarray[double, ndim=1] a)
    cdef np.ndarray[double, ndim=1] fit_transform(self, double[:] X)
    cdef bytes get_wisdom(self)