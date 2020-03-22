import sys
from setuptools import setup, Extension ,find_packages

print("FIND PACKAGE",find_packages())

setup(
    name="simweights",
    version = "0.0.1",
    packages=find_packages(),
    keywords = [
        'neutrino', 'cosmic rays',
    ],

    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],   
)
    
