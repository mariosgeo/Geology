"""Setup for Simple Lib"""
from vtkclass import __version__


from setuptools import (
    setup,
    find_packages
)

setup(
    name='geovtk',
    version=__version__,
    description='Python library for Generatinhg vtk files from geological and geophysical data',
    author='Marios Karaoulis',
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "log",
            "log.*",
            "*.log",
            "*.log.*"
        ]
    ),
    install_requires=[
        'numpy',
        'pandas',
        'glob',
        'gdal'
    ]
)
