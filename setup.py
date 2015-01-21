#!/usr/bin/env python
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import platform

if platform.system() == 'Windows':
    fftwlib = 'libfftw3-3.dll'
else:
    fftwlib = 'fftw3'

random_projection = Extension("random_projection_fast",
                sources=["fjlt/random_projection_fast.pyx"],
                include_dirs=[numpy.get_include()],
                libraries=[fftwlib])

srht = Extension("SubsampledRandomizedFourrierTransform1d",
                sources=["fjlt/SubsampledRandomizedFourrierTransform1d.pyx", "fjlt/SubsampledRandomizedFourrierTransform1d.pxd"],
                include_dirs=[numpy.get_include()],
                libraries=[fftwlib])

setup(ext_modules=[random_projection, srht],
 cmdclass={'build_ext': build_ext})

setup(name='FJLT',
      version='1.1',
      description='Fast Johnson Lindenstrauss Transform',
      author='Gabriel Krummenacher',
      author_email='gabriel.krummenacher@inf.ethz.ch',
      url='http://people.inf.ethz.ch/kgabriel/software.html',
      packages=['fjlt'],
      py_modules=['example'],
      requires=['numpy'],
      ext_modules=[random_projection, srht],
      cmdclass={'build_ext': build_ext}
     )
