"""
Fast Johnson-Lindenstrauss Transform (FJLT)

http://people.inf.ethz.ch/kgabriel/software.html
"""


import numpy as np
from fjlt.srht import SubsampledRandomizedHadamardTransform

def demo():
    X = np.asfortranarray(np.random.randn(1000, 100))
    srht = SubsampledRandomizedHadamardTransform(600)
    Y = srht.fit_transform(X)

    sigma_X = np.linalg.norm(X, 2)
    sigma_Y = np.linalg.norm(Y, 2)
    print(np.abs(sigma_X - sigma_Y) / sigma_Y)

if __name__ == '__main__':
    demo()
