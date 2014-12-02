"""
Fast Johnson-Lindenstrauss Transform (FJLT)

http://people.inf.ethz.ch/kgabriel/software.html
"""

from __future__ import division
import numpy as np
from random_projection_fast import fast_unitary_transform_fast, fast_unitary_transform_fast_1d, \
    inverse_fast_unitary_transform_fast_1d

class SubsampledRandomizedFourrierTransform(object):
    def __init__(self, k, rows=True):
        self.rows = rows
        self.k = k

    def fit(self, X, y=None):
        assert (y is None) or self.rows, 'If over features, cant use y'
        if not self.rows:
            X = X.T

        self.n = X.shape[0]
        self.D = np.sign(np.random.randn(self.n))
        self.srht_const = np.sqrt(self.n / self.k)
        self.S = np.random.choice(self.n, self.k, replace=False)

    def transform_1d(self, x):
        a = np.asarray(fast_unitary_transform_fast_1d(x, D=self.D))
        return self.srht_const * a[self.S]

    def inverse_transform_1d(self, a):
        x = np.zeros(self.n)
        x[self.S] = a / self.srht_const
        return inverse_fast_unitary_transform_fast_1d(x, D=self.D)

    def transform(self, X, y=None):
        assert (y is None) or self.rows, 'If over features, cant use y'

        if y is not None:
            X = np.c_[X, y]
            Ab = fast_unitary_transform_fast(X, D=self.D)
            return self.srht_const * Ab[self.S, 0:-1], self.srht_const * Ab[self.S, -1]
        else:
            if self.rows:
                A = fast_unitary_transform_fast(X, D=self.D)
                return self.srht_const * A[self.S, :]
            else:
                A = fast_unitary_transform_fast(X.T, D=self.D).T
                return self.srht_const * A[:, self.S]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

#     def inverse_transform(self, A):
#         inv_srht_const = 1/self.srht_const
#         if self.rows:
#             k, d = A.shape
#             X = np.zeros((self.n, d), order='F')
#             X[self.S, :] = inv_srht_const * A
#             return inverse_fast_unitary_transform_fast(X, self.D)
#         else:
#             n, k = A.shape
#             X = np.zeros((n, self.n), order='C')
#             X[:, self.S] = inv_srht_const * A
#             return inverse_fast_unitary_transform_fast(X.T, self.D).T

def test_1d():
    n, d = 1, 10
    k = 5

    np.random.seed(1)
    X = np.random.random_sample((n, d))
    srft = SubsampledRandomizedFourrierTransform(k)
    srft.fit(X[0, :])

    print X.T
    print X[0, :][:, None]
    print X[0, :]
    print ''

    print np.squeeze(srft.transform(np.asfortranarray(X.copy().T)).T)
    print np.squeeze(srft.transform(np.asfortranarray(X[0, :][:, None])))
    print srft.transform_1d(X[0, :])

def test_inverse_1d():
    n, d = 1, 10
    k = 10

    np.random.seed(1)
    X = np.random.random_sample((n, d))
    srft = SubsampledRandomizedFourrierTransform(k)
    srft.fit(X[0, :])
    a = srft.transform_1d(X[0, :].copy())
    x_app = srft.inverse_transform_1d(a)

    print np.c_[X.T, x_app]