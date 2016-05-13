"""
Fast Johnson-Lindenstrauss Transform (FJLT)

http://people.inf.ethz.ch/kgabriel/software.html
"""

from __future__ import division

from random_projection_fast import fast_unitary_transform_fast, fast_unitary_transform_fast_1d, \
    inverse_fast_unitary_transform_fast_1d, import_wisdom, export_wisdom, fast_unitary_transform_fast_1d32

import numpy as np


class SubsampledRandomizedFourrierTransform(object):
    def __init__(self, k, rows=True, wisdom_file=None, floatX='float64'):
        self.rows = rows
        self.k = k
        self.floatX = floatX
        
        if wisdom_file is None:
            wisdom_file = 'srft_wisdom'

        self.wisdom_file = wisdom_file
        try:
            import_wisdom(self.wisdom_file)
        except IOError:
            print 'wisdom file', self.wisdom_file, 'not found, starting new file.'

    def fit(self, X, y=None, prefit=False):
        assert (y is None) or self.rows, 'If over features, cant use y'
        if not self.rows:
            X = X.T

        self.n = X.shape[0]
        self.D = np.sign(np.random.randn(self.n))
        self.srht_const = np.sqrt(self.n / self.k)
        self.S = np.random.choice(self.n, self.k, replace=False)
        if self.floatX == 'float32':
            self.D = self.D.astype(np.float32)

        if prefit:
            if len(X.shape) == 1:
                self.transform_1d(X)
            else:
                self.transform(X, y)

            export_wisdom(self.wisdom_file)

    def transform_1d(self, x):
        if self.floatX == 'float32':
            a = np.asarray(fast_unitary_transform_fast_1d32(x.copy(), D=self.D), dtype=np.float32)
        else:
            a = np.asarray(fast_unitary_transform_fast_1d(x.copy(), D=self.D))
        return self.srht_const * a[self.S]

    def inverse_transform_1d(self, a):
        x = np.zeros(self.n)
        x[self.S] = a  # / self.srht_const
        return np.asarray(inverse_fast_unitary_transform_fast_1d(x, D=self.D))

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

    def __del__(self):
        export_wisdom(self.wisdom_file)

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
    k = 8

    np.random.seed(3)
    X = np.random.random_sample((n, d))
    srft = SubsampledRandomizedFourrierTransform(k)
    srft.fit(X[0, :])
    a = srft.transform_1d(X[0, :])
    x_app = srft.inverse_transform_1d(a)

    print np.c_[X.T, x_app]
    print np.linalg.norm(X), np.linalg.norm(x_app)
