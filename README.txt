Fast Johnson-Lindenstrauss Transform (FJLT)

==========================================

Cython and python implementation of the Fast JLT. Uses FFTW.


1. Dependencies
---------------
- FFTW, a fast DFT library, available from: http://www.fftw.org/download.html.
- Cython


2. Install
----------

python setup.py install


3. Usage
--------

From python use the class SubsampledRandomizedHadamardTransform in srht.py.
It follows the scikit-learn interface.
From cython use SubsampledRandomizedHadamardTransform1d in
SubsampledRandomizedHadamardTransform1d.pxy or the functions in
random_projection_fast.pyx directly.

See demo.py for a quick demo.

4. Authors / Contact
--------------------
Gabriel Krummenacher

http://people.inf.ethz.ch/kgabriel/software.html
