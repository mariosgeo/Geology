# Geology - Professional Geological Model Reconstruction Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/geology-Marios-toolkit.svg)](https://badge.fury.io/py/geology-Marios-toolkit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/geology-Marios-toolkit)](https://pypistats.org/packages/geology-Marios-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/geology/badge/?version=latest)](https://geology.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **📦 Package Status: Successfully published on Test PyPI as `geology-Marios-toolkit`**  
> **🚀 Installation:** `pip install --index-url https://test.pypi.org/simple/ geology-Marios-toolkit`

> **A comprehensive Python toolkit for geological model reconstruction using advanced inpainting techniques and machine learning methods.**

This repository contains a professional implementation for geological subsurface model reconstruction from sparse data using state-of-the-art computational methods. The toolkit combines biharmonic inpainting, machine learning classification, and uncertainty quantification to create robust geological models from incomplete datasets.

![Geological Model](images/figure_2.svg)

## 🧬 Scientific Overview

Geological subsurface characterization often faces the challenge of sparse and irregularly distributed data points. This toolkit addresses this fundamental problem by implementing advanced computational methods for geological model reconstruction:

- **🔬 Biharmonic Inpainting**: Smooth interpolation preserving geological boundaries and structural continuity
- **🤖 Machine Learning Classification**: One-vs-all and probabilistic classification for multi-class geological units  
- **📊 Weighted Interpolation**: Anisotropic interpolation respecting geological fabric and preferential directions
- **📈 Uncertainty Quantification**: Comprehensive uncertainty analysis with confidence intervals and error propagation
- **🎯 3D Visualization**: Professional VTK-based visualization for geological models and validation

## 🚀 Key Features

### Core Functionality
- **Advanced Inpainting Algorithms**: Biharmonic PDE-based interpolation for geological boundaries
- **Multi-Class Classification**: Sophisticated geological unit prediction with uncertainty estimates
- **Anisotropic Interpolation**: Directional interpolation respecting geological structures
- **Memory-Efficient Processing**: Batch processing for large geological datasets
- **Cross-Platform Compatibility**: Windows, Linux, and macOS support

### Data Integration
- **Multiple Data Formats**: Support for borehole logs, geological surveys, and geophysical data
- **Geospatial Integration**: Native support for coordinate systems and geospatial data formats
- **Quality Control**: Automated data validation and outlier detection
- **Missing Data Handling**: Robust algorithms for incomplete geological datasets

### Visualization and Export
- **Professional 3D Visualization**: High-quality geological model rendering
- **Publication-Ready Figures**: Scientific plotting with geological colormaps and annotations
- **VTK Export**: Compatible with ParaView, VisIt, and other professional visualization software
- **Interactive Dashboards**: Jupyter notebook integration with interactive widgets

## 📁 Project Structure

```
Geology/
├── 📓 demo.ipynb                    # Main demonstration notebook with examples
├── 📋 requirements.txt              # Core dependencies
├── 🏗️ setup.py                     # Professional package configuration
├── 📖 README.md                     # This comprehensive guide
├── 📊 Data Files/
│   ├── geotop.npy                   # 3D geological model data
│   ├── top_layer.gpkg              # Geological layer (GeoPackage format)
│   ├── data_final.xlsx             # Processed geological dataset
│   └── real_model.npy              # Reference geological model
├── 🧮 gridder/                     # Geological gridding and inpainting
│   ├── __init__.py                 # Module initialization
│   └── gridder.py                  # Core geological algorithms
├── 🎨 geo_vtk/                     # Professional VTK visualization tools
│   ├── src/vtkclass/               # VTK conversion classes
│   ├── data/                       # Example geological datasets
│   └── README.md                   # VTK toolkit documentation
├── 🖼️ images/                      # Documentation figures and results
├── 📚 docs/                        # Comprehensive documentation
└── 🧪 tests/                       # Automated testing suite
```

## 📦 Installation

### PyPI Package Available!

**Geology** is now available on the Python Package Index (PyPI), making installation as simple as:

```bash
pip install geology-Marios-toolkit
```

This command installs the complete geological toolkit with all core dependencies. For specialized applications, optional dependencies are available through extras (see installation options below).

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**: Modern Python with scientific computing support
- **Jupyter Notebook**: For interactive geological modeling workflows

### Installation Options

#### Option 1: PyPI Installation (Recommended)
```bash
# Install from PyPI (simplest method)
pip install geology-Marios-toolkit

# Launch interactive notebook
jupyter notebook demo.ipynb
```

#### Option 2: PyPI with Enhanced Features
```bash
# Install with visualization enhancements
pip install geology-Marios-toolkit[visualization]

# Install with geospatial capabilities  
pip install geology-Marios-toolkit[geospatial]

# Install with development tools
pip install geology-Marios-toolkit[dev]

# Install complete toolkit with all features
pip install geology-Marios-toolkit[all]
```

#### Option 3: Test PyPI Installation (For Testing)
```bash
# Install from Test PyPI (currently available)
pip install --index-url https://test.pypi.org/simple/ geology-Marios-toolkit

# Note: Test PyPI installation may have dependency issues
# as not all dependencies are available on Test PyPI
```

#### Option 4: Development Installation from Source
```bash
# Clone the repository
git clone https://github.com/mariosgeo/Geology.git
cd Geology

# Install in development mode
pip install -e .

# Or install with development tools
pip install -e .[dev]
```

#### Option 5: Manual Installation from Source
```bash
# Clone and set up from source
git clone https://github.com/mariosgeo/Geology.git
cd Geology

# Install core dependencies
pip install -r requirements.txt

# Install package
pip install .
```

### Quick Example

```python
import geology
import numpy as np

# Create geological gridder
geo_model = geology.create_geological_model()

# Set up geological grid
geo_model.make_grid(dx=1.0, dy=1.0)  # 1m resolution

# Load borehole data and perform gridding
geo_model.gridder()

# Perform geological inpainting
geo_model.one_vs_all(x_weight=1.0, y_weight=3.0)  # Anisotropic weights

# Create 3D visualization
vtk_converter = geology.create_vtk_converter()
vtk_converter.make_3d_grid_to_vtk('geological_model.vtk', 
                                  geo_model.prediction_data,
                                  x_coords, y_coords, z_coords)

print(f"Geology package version: {geology.get_version()}")
print(f"Geological model created with {geo_model.uncertainty:.2%} average uncertainty")
```

## 🛠️ Package Development Status

### ✅ **COMPLETED** (Package is Built and Available):
- ✅ Package configuration (`setup.py`, `pyproject.toml`)
- ✅ Distribution files created and tested
- ✅ Successfully uploaded to Test PyPI
- ✅ Package name: `geology-Marios-toolkit`

### 🚀 **Current Package Access**:

**Test PyPI (Available Now):**
```bash
pip install --index-url https://test.pypi.org/simple/ geology-Marios-toolkit
```

**Production PyPI (Coming Soon):**
```bash
pip install geology-Marios-toolkit
```

### 📝 **Package Information**:
- **Package Name**: `geology-Marios-toolkit`
- **Version**: 1.0.0
- **Test PyPI URL**: https://test.pypi.org/project/geology-Marios-toolkit/
- **Dependencies**: Full scientific Python stack included

---

**🌍 Advancing Geological Understanding Through Machine Learning 🌍**

[![Made with ❤️ for Geoscience](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20for-Geoscience-blue)](https://github.com/mariosgeo/Geology)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Test PyPI](https://img.shields.io/badge/Test%20PyPI-Available-brightgreen)](https://test.pypi.org/project/geology-Marios-toolkit/)
[![VTK](https://img.shields.io/badge/VTK-3D%20Visualization-green)](https://vtk.org)
[![Open Science](https://img.shields.io/badge/Open-Science-orange)](https://github.com/mariosgeo/Geology)

**Now available on Test PyPI:** `pip install --index-url https://test.pypi.org/simple/ geology-Marios-toolkit`