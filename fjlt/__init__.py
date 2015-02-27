import os
from .version import __version__

def get_include():
    ''' Path of cython headers for compiling cython modules '''
    return os.path.dirname(os.path.abspath(__file__))
