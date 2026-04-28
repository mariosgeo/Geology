"""
Professional setup configuration for GeoVTK - Geological VTK Data Visualization Library

A comprehensive Python package for converting geological and geophysical data 
to VTK format for 3D visualization and analysis.
"""

import os
from setuptools import setup, find_packages

# Read README for long description
def read_readme():
    """Read README file for package description."""
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "GeoVTK - Professional geological data visualization toolkit"

# Package version
__version__ = "1.0.0"

# Core dependencies for geological data processing
INSTALL_REQUIRES = [
    'numpy>=1.18.0',           # Core numerical computing
    'pandas>=1.0.0',           # Data manipulation and analysis
    'matplotlib>=3.0.0',       # Scientific plotting and colormaps
    'scipy>=1.4.0',            # Scientific computing and interpolation
    'GDAL>=3.0.0',             # Geospatial data abstraction library
]

# Optional dependencies for enhanced functionality
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=6.0.0',        # Testing framework
        'pytest-cov>=2.10.0',   # Coverage testing
        'black>=21.0.0',        # Code formatting
        'flake8>=3.8.0',        # Linting
        'sphinx>=3.0.0',        # Documentation generation
    ],
    'visualization': [
        'mayavi>=4.7.0',        # Advanced 3D visualization
        'pyvista>=0.30.0',      # Modern VTK interface
        'vtk>=9.0.0',           # VTK library
    ],
    'geospatial': [
        'geopandas>=0.8.0',     # Geospatial data analysis
        'rasterio>=1.2.0',      # Raster data I/O
        'pyproj>=3.0.0',        # Cartographic projections
    ]
}

# All optional dependencies
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    # Package metadata
    name='geovtk',
    version=__version__,
    
    # Package description
    description='Professional Python library for geological and geophysical VTK data visualization',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    # Author and contact information
    author='Marios Karaoulis',
    author_email='marios.karaoulis@example.com',  # Update with actual email
    maintainer='Marios Karaoulis',
    maintainer_email='marios.karaoulis@example.com',
    
    # Project URLs
    url='https://github.com/mariosgeo/Geology',
    project_urls={
        'Documentation': 'https://github.com/mariosgeo/Geology/wiki',
        'Source': 'https://github.com/mariosgeo/Geology',
        'Tracker': 'https://github.com/mariosgeo/Geology/issues',
    },
    
    # Package discovery and content
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*", 
            "tests.*",
            "tests",
            "log",
            "log.*",
            "*.log",
            "*.log.*",
            "demos.*",
            "examples.*"
        ]
    ),
    
    # Dependencies
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Package classification
    classifiers=[
        # Development status
        'Development Status :: 4 - Beta',
        
        # Intended audience
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education', 
        'Intended Audience :: Developers',
        
        # Topic classification
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        
        # License
        'License :: OSI Approved :: MIT License',
        
        # Programming language
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        
        # Operating systems
        'Operating System :: OS Independent',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
    ],
    
    # Keywords for package discovery
    keywords=[
        'geology', 'geophysics', 'vtk', 'visualization', '3d-modeling',
        'geological-data', 'borehole', 'resistivity', 'geospatial',
        'earth-science', 'geological-modeling', 'data-visualization'
    ],
    
    # Package data and resources
    include_package_data=True,
    package_data={
        'geovtk': [
            'data/*.txt',
            'data/*.json', 
            'data/colormaps/*.txt',
            'examples/*.py',
        ],
    },
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'geovtk-convert=geovtk.cli:main',
        ],
    },
    
    # Zip safety
    zip_safe=False,
)
