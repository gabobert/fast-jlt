"""
Fast Johnson-Lindenstrauss Transform (FJLT)

http://people.inf.ethz.ch/kgabriel/software.html
"""

from __future__ import division

cimport cython
cimport numpy as np
import numpy as np
from numpy.random import randn
from numpy.random import choice
#from profilehooks import profile

cdef extern from "math.h":
    double sqrt(double x)

cdef extern from "fftw3.h":
    ctypedef struct fftw_plan:
        pass
    cdef enum fftw_r2r_kind_do_not_use_me:
        FFTW_R2HC=0
        FFTW_HC2R=1
        FFTW_DHT=2
        FFTW_REDFT00=3
        FFTW_REDFT01=4
        FFTW_REDFT10=5
        FFTW_REDFT11=6
        FFTW_RODFT00=7
        FFTW_RODFT01=8
        FFTW_RODFT10=9
        FFTW_RODFT11=10
    
    cdef enum:
        FFTW_MEASURE = 0
        FFTW_DESTROY_INPUT = 1
        FFTW_UNALIGNED = 2
        FFTW_CONSERVE_MEMORY = 4
        FFTW_EXHAUSTIVE = 8
        FFTW_PRESERVE_INPUT = 16
        FFTW_PATIENT = 32
        FFTW_ESTIMATE = 64
        FFTW_WISDOM_ONLY = 2097152

    ctypedef fftw_r2r_kind_do_not_use_me fftw_r2r_kind
    fftw_plan fftw_plan_r2r(int rank, const int *n, double *input, double *output, const fftw_r2r_kind *kind, unsigned int flags)
    fftw_plan fftw_plan_many_r2r(int rank, const int *n, int howmany,
                                  double *input, const int *inembed,
                                  int istride, int idist,
                                  double *output, const int *onembed,
                                  int ostride, int odist,
                                  const fftw_r2r_kind *kind, unsigned flags)
    void fftw_execute_r2r(const fftw_plan p, double *input, double *output)
    void fftw_execute(const fftw_plan p)
    char *fftw_export_wisdom_to_string()
    int fftw_import_wisdom_from_string(const char *input_string)
    
    fftw_plan fftwf_plan_r2r(int rank, const int *n, float *input, float *output, const fftw_r2r_kind *kind, unsigned int flags)
    void fftwf_execute_r2r(const fftw_plan p, float *input, float *output)

cpdef import_wisdom(wisdom_file):
    with open(wisdom_file, 'rb') as wsdf:
        wisdom = wsdf.read()
        fftw_import_wisdom_from_string(wisdom)


cpdef export_wisdom(wisdom_file):
    wisdom = fftw_export_wisdom_to_string()
    print wisdom
    with open(wisdom_file, 'wb') as wsdf:
        wsdf.write(wisdom)

def import_wisdom_from_string(const char *input_string):
    return fftw_import_wisdom_from_string(input_string)

def export_wisdom_to_string():
    return fftw_export_wisdom_to_string()


def precompute_plan_many(np.ndarray[double, ndim=2, mode='c'] X):
    cdef np.int_t n = X.shape[0]
    cdef np.int_t d = X.shape[1]
    cdef int howmany = d
    cdef int istride = howmany  # 1
    cdef int ostride = istride
    cdef int idist = 1  # n
    cdef int odist = idist
    cdef np.ndarray[int, ndim=1] dims = np.asarray([n], dtype=np.intc)
    cdef np.ndarray[double, ndim=2] plan_arr = np.empty((n,d), dtype=float)
    cdef fftw_r2r_kind kind = FFTW_REDFT10

    cdef fftw_plan fftplan = fftw_plan_many_r2r(1, &dims[0], howmany, &X[0,0], &dims[0], istride, idist, &X[0,0], &dims[0], ostride, odist, &kind, 8)

    return fftplan



@cython.boundscheck(False)
@cython.wraparound(False)
def fast_unitary_transform_fast_many(np.ndarray[double, ndim=2, mode='c'] X, np.ndarray[double, ndim=1] D, fftw_plan fftplan):

    cdef np.int_t n = X.shape[0]
    cdef np.int_t d = X.shape[1]

    cdef double scale = 1 / np.sqrt(2 * n)

    for i in range(n):
        X[i, :] = scale * D[i] * X[i, :]

    fftw_execute(fftplan)


    X[0, :] /= np.sqrt(2)

    return X

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_unitary_transform_fast(np.ndarray[double, ndim=2, mode='fortran'] X, np.ndarray[double, ndim=1] D):

    cdef np.int_t n = X.shape[0]
    cdef np.int_t d = X.shape[1]

    cdef double scale = 1 / np.sqrt(2 * n)

    for i in range(n):
        X[i, :] = scale * D[i] * X[i, :]

#     lib.fftw_init_threads()
#     lib.fftw_plan_with_nthreads(4)
    cdef np.ndarray[int, ndim=1] dims = np.asarray([n], dtype=np.intc)
    cdef np.ndarray[double, ndim=1] plan_arr = np.empty(n, dtype=float)
    cdef fftw_r2r_kind kind = FFTW_REDFT10
    cdef fftw_plan fftplan = fftw_plan_r2r(1, &dims[0], &plan_arr[0], &plan_arr[0], &kind, 8)
#     fftplan = Plan(input_array, input_array, realtypes=["realeven 10"], flags=['exhaustive'])  # 'destroy input'

#     print 'X after random signs, before fft', X

    for i in range(d):
        fftw_execute_r2r(fftplan, &X[0, i], &X[0, i])

#     print 'X after fft, before first row scaling', X

    X[0, :] /= np.sqrt(2)

    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] inverse_fast_unitary_transform_fast_1d(double[:] A, double[:] D):

    cdef np.int_t n = A.shape[0]

    A[0] *= sqrt(2)

    cdef int[:] dims = np.asarray([n], dtype=np.intc)
    cdef double[:] plan_arr = np.empty(n, dtype=float)
    cdef fftw_r2r_kind kind = FFTW_REDFT01
    cdef fftw_plan fftplan = fftw_plan_r2r(1, &dims[0], &plan_arr[0], &plan_arr[0], &kind, 8)


    fftw_execute_r2r(fftplan, &A[0], &A[0])

    cdef double scale = 1 / sqrt(2 * n)

    for i in range(n):
        A[i] = scale * D[i] * A[i]


    return A

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] fast_unitary_transform_fast_1d(double[:] X, double[:] D):

    cdef np.int_t n = X.shape[0]

    cdef double scale = 1 / sqrt(2 * n)

    for i in range(n):
        X[i] = scale * D[i] * X[i]

    cdef int[:] dims = np.asarray([n], dtype=np.intc)
    cdef double[:] plan_arr = np.empty(n, dtype=float)
    cdef fftw_r2r_kind kind = FFTW_REDFT10
    cdef fftw_plan fftplan = fftw_plan_r2r(1, &dims[0], &plan_arr[0], &plan_arr[0], &kind, 8)

    fftw_execute_r2r(fftplan, &X[0], &X[0])

    X[0] /= sqrt(2)

    return X

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fftw_plan fast_unitary_transform_fast_1d32_plan(float[:] X, char* wisdom_file):
    cdef np.int_t n = X.shape[0]
    cdef int[:] dims = np.asarray([n], dtype=np.intc)
#     cdef float[:] plan_arr = np.empty(n, dtype=np.float32)
    cdef fftw_r2r_kind kind = FFTW_REDFT10
    export_wisdom(wisdom_file)
    cdef fftw_plan fftplan = fftwf_plan_r2r(1, &dims[0], &X[0], &X[0], &kind, FFTW_EXHAUSTIVE)
    print fftplan
    
    export_wisdom(wisdom_file)
    
    return fftplan


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:] fast_unitary_transform_fast_1d32_wsdmonly(float[:] X, float[:] D):

    cdef np.int_t n = X.shape[0]

    cdef float scale = 1 / sqrt(2 * n)

    for i in range(n):
        X[i] = scale * D[i] * X[i]

    cdef int[:] dims = np.asarray([n], dtype=np.intc)
#     cdef float[:] plan_arr = np.empty(n, dtype=np.float32)
    cdef fftw_r2r_kind kind = FFTW_REDFT10
    cdef fftw_plan fftplan = fftwf_plan_r2r(1, &dims[0], &X[0], &X[0], &kind, FFTW_WISDOM_ONLY)


    fftwf_execute_r2r(fftplan, &X[0], &X[0])

    X[0] /= sqrt(2)

    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:] fast_unitary_transform_fast_1d32(float[:] X, float[:] D):
#     print export_wisdom_to_string()
    cdef np.int_t n = X.shape[0]

    cdef float scale = 1 / sqrt(2 * n)

    for i in range(n):
        X[i] = scale * D[i] * X[i]

    cdef int[:] dims = np.asarray([n], dtype=np.intc)
    cdef float[:] plan_arr = np.empty(n, dtype=np.float32)
    cdef fftw_r2r_kind kind = FFTW_REDFT10
    cdef fftw_plan fftplan = fftwf_plan_r2r(1, &dims[0], &plan_arr[0], &plan_arr[0], &kind, FFTW_ESTIMATE)

#     print fftplan

    fftwf_execute_r2r(fftplan, &X[0], &X[0])

    X[0] /= sqrt(2)

    return X

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] fast_unitary_transform_fast_1d_planned(double[:] X, double[:] A, double[:] D, fftw_plan fftplan):

    cdef np.int_t n = X.shape[0]

    cdef double scale = 1 / sqrt(2 * n)

    for i in range(n):
        X[i] = scale * D[i] * X[i]

    fftw_execute_r2r(fftplan, &X[0], &A[0])

    A[0] /= sqrt(2)

    return A

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] inverse_fast_unitary_transform_fast_1d_planned(double[:] A, double[:] X, double[:] D, fftw_plan fftplan):

    cdef np.int_t n = A.shape[0]

    A[0] *= sqrt(2)


    fftw_execute_r2r(fftplan, &A[0], &X[0])

    cdef double scale = 1 / sqrt(2 * n)

    for i in range(n):
        X[i] = scale * D[i] * X[i]


    return X
