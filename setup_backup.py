"""
Professional setup configuration for Geology - Geological Model Inpainting

A comprehensive research and development package for geological mo    ],
    
    # Package data and resources - include all data files
    include_package_data=True,
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

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

# Package version
__version__ = "1.0.0"

# Core dependencies from requirements.txt
INSTALL_REQUIRES = read_requirements()

# Optional dependencies for enhanced functionality
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=6.0.0',        # Testing framework
        'pytest-cov>=2.10.0',   # Coverage testing
        'black>=21.0.0',        # Code formatting
        'flake8>=3.8.0',        # Linting
        'sphinx>=3.0.0',        # Documentation generation
        'jupyter>=1.0.0',       # Jupyter notebook support
        'nbformat>=5.0.0',      # Notebook format support
    ],
    'visualization': [
        'mayavi>=4.7.0',        # Advanced 3D visualization
        'pyvista>=0.30.0',      # Modern VTK interface
        'plotly>=5.0.0',        # Interactive plotting
        'seaborn>=0.11.0',      # Statistical visualization
    ],
    'geospatial': [
        'rasterio>=1.2.0',      # Raster data I/O
        'pyproj>=3.0.0',        # Cartographic projections
        'folium>=0.12.0',       # Web mapping
        'contextily>=1.2.0',    # Basemap tiles
    ],
    'ml': [
        'tensorflow>=2.8.0',    # Deep learning
        'pytorch>=1.11.0',      # Alternative deep learning
        'xgboost>=1.6.0',       # Gradient boosting
        'lightgbm>=3.3.0',      # Light gradient boosting
    ]
}

# All optional dependencies
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    # Package metadata
    name='Geology',
    version=__version__,
    
    # Package description
    description='Professional geological model reconstruction using advanced inpainting techniques',
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
        'Research Paper': 'https://doi.org/10.1000/example',  # Update with actual DOI
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
        # Development status
        'Development Status :: 4 - Beta',
        
        # Intended audience
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        
        # Topic classification
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        
        # License
        'License :: OSI Approved :: MIT License',
        
        # Programming language
        'Programming Language :: Python :: 3',
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
        'geology', 'geophysics', 'inpainting', 'machine-learning',
        'geological-modeling', 'biharmonic-interpolation', 'subsurface',
        'borehole-data', 'geological-reconstruction', 'earth-science',
        'spatial-analysis', 'geological-uncertainty', 'one-vs-all'
    ],
    
    # Package data and resources
    include_package_data=True,
    package_data={
        '': [
            '*.txt',
            '*.md',
            '*.yml',
            '*.yaml',
            'data/*.npy',
            'data/*.csv',
            'data/*.xlsx',
            'images/*.svg',
            'images/*.png',
            'images/*.jpg',
        ],
    },
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'geology-inpaint=geology.cli:main',
            'geo-reconstruct=geology.reconstruction:main',
        ],
    },
    
    # Zip safety
    zip_safe=False,
    
    # Project maturity
    project_maturity='beta',
)