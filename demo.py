"""
Fast Johnson-Lindenstrauss Transform (FJLT)

http://people.inf.ethz.ch/kgabriel/software.html
"""


import numpy as np
from time import time
from fjlt.SubsampledRandomizedFourrierTransform import SubsampledRandomizedFourrierTransform, test_inverse_1d
from fjlt.demo_cython import demo_cython


def demo_1d():
    x = np.random.randn(1000)
    srft = SubsampledRandomizedFourrierTransform(500)
    tic = time()
    srft.fit(x, prefit=True)
    print 'Prefit:', time() - tic
    tic = time()
    y = srft.transform_1d(x)
    print 'Transform:', time() - tic

def demo():
    X = np.asfortranarray(np.random.randn(1000, 100))
    srft = SubsampledRandomizedFourrierTransform(600)
    Y = srft.fit_transform(X)

    sigma_X = np.linalg.norm(X, 2)
    sigma_Y = np.linalg.norm(Y, 2)
    print(np.abs(sigma_X - sigma_Y) / sigma_Y)

def demo_I():
    k, d = 2, 11
#     X = np.eye(d)
    nrm_X, nrm_Y2 = 0, 0
    for i in range(100):
        X = np.eye(d)
        srft = SubsampledRandomizedFourrierTransform(k)
        srft.fit(X)
        Y1 = srft.transform(np.asfortranarray(X.copy()))
        Y2 = srft.transform(np.asfortranarray(Y1.copy().T)).T
#     print X
#     print Y2
#     print srft.srht_const ** 2

#     print np.linalg.norm(X[0, :]), np.linalg.norm(Y2[0, :])
#     print np.linalg.norm(X[:, 0]), np.linalg.norm(Y2[:, 0])

        nrm_X += np.linalg.norm(X)
        nrm_Y2 += np.linalg.norm(Y2)

    print nrm_X / 100, nrm_Y2 / 100

if __name__ == '__main__':
    demo_I()
    exit()

    demo_1d()
    demo()
    demo_cython()
    test_inverse_1d()
