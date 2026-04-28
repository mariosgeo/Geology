"""
Professional setup configuration for Geology - Geological Model Inpainting

A comprehensive research and development package for geological model reconstruction
using advanced inpainting techniques and machine learning methods.
"""

import os
from setuptools import setup, find_packages

# Read README for long description
def read_readme():
    """Read README file for package description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Geology - Professional geological model inpainting toolkit"

# Package version
__version__ = "1.0.0"

# Core dependencies
INSTALL_REQUIRES = [
    'numpy>=1.19.0',
    'scipy>=1.7.0',
    'matplotlib>=3.3.0',
    'scikit-learn>=0.24.0',
    'scikit-image>=0.18.0',
    'pandas>=1.3.0',
    'geopandas>=0.9.0',
    'vtk>=9.0.0',
    'gdal>=3.2.0',
    'jupyter>=1.0.0',
]

# Optional dependencies for enhanced functionality
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'black>=21.0.0',
        'flake8>=3.8.0',
        'sphinx>=3.0.0',
    ],
    'visualization': [
        'mayavi>=4.7.0',
        'pyvista>=0.30.0',
        'plotly>=5.0.0',
        'seaborn>=0.11.0',
    ],
    'geospatial': [
        'rasterio>=1.2.0',
        'pyproj>=3.0.0',
        'folium>=0.12.0',
        'contextily>=1.2.0',
    ],
}

# All optional dependencies
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    # Package metadata
    name="Geology",
    version=__version__,
    
    # Package description
    description='Professional geological model reconstruction using advanced inpainting techniques',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    # Author and contact information
    author='Marios Karaoulis',
    author_email='marios.karaoulis@example.com',
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
    packages=find_packages(include=['geology', 'geology.*']),
    include_package_data=True,
    
    # Dependencies
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Package classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords for discoverability
    keywords=[
        'geology', 'geophysics', 'inpainting', 'machine-learning',
        'geological-modeling', 'biharmonic-interpolation', 'subsurface',
        'borehole-data', 'geological-reconstruction', 'earth-science',
        'spatial-analysis', 'geological-uncertainty', 'one-vs-all'
    ],
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'geology-inpaint=geology.core:main',
            'geo-model=geology.core:create_geological_model',
            'geo-vtk=geology.core:create_vtk_converter',
        ],
    },
    
    # Package metadata for academic use
    zip_safe=False,
    platforms=['any'],
)