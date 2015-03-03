#!/usr/bin/env python
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import platform
from Cython.Build.Dependencies import cythonize

if platform.system() == 'Windows':
    fftwlib = 'libfftw3-3.dll'
else:
    fftwlib = 'fftw3'

random_projection = Extension("fjlt.random_projection_fast",
                sources=["fjlt/random_projection_fast.pyx"],
                include_dirs=[numpy.get_include()],
                libraries=[fftwlib])

srft = Extension("fjlt.SubsampledRandomizedFourrierTransform1d",
                sources=["fjlt/SubsampledRandomizedFourrierTransform1d.pyx", "fjlt/SubsampledRandomizedFourrierTransform1d.pxd"],
                include_dirs=[numpy.get_include()],
                libraries=[fftwlib])

demo = Extension("fjlt.demo_cython",
                sources=["demo_cython.pyx"],
                include_dirs=[numpy.get_include()],
                libraries=[fftwlib])

# setup(ext_modules=[random_projection, srft],
#  cmdclass={'build_ext': build_ext})

exec(open('fjlt/version.py').read())

setup(name='FJLT',
      version=__version__,
      description='Fast Johnson Lindenstrauss Transform',
      author='Gabriel Krummenacher',
      author_email='gabriel.krummenacher@inf.ethz.ch',
      url='http://people.inf.ethz.ch/kgabriel/software.html',
      packages=['fjlt'],
#       py_modules=['demo'],
      requires=['numpy'],
      ext_modules=cythonize([random_projection, srft, demo]),
      cmdclass={'build_ext': build_ext}
     )
