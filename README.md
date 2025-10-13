# Geology - Professional Geological Model Reconstruction Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/Geology.svg)](https://badge.fury.io/py/Geology)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/Geology)](https://pypistats.org/packages/geology)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/geology/badge/?version=latest)](https://geology.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **📦 Package Status: ✅ LIVE ON PYPI! 🎉**  
> **🚀 Installation:** `pip install Geology`  
> **🌐 PyPI URL:** https://pypi.org/project/Geology/

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

### 🎉 Now Available on PyPI!

**Geology** is officially published on the Python Package Index (PyPI)! Installation is now as simple as:

```bash
pip install Geology
```

This command installs the complete geological toolkit with all core dependencies. The package is now globally available to the Python community for geological modeling and analysis.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**: Modern Python with scientific computing support
- **Jupyter Notebook**: For interactive geological modeling workflows

### Installation Options

#### Option 1: PyPI Installation (Recommended) ✅
```bash
# Install from PyPI - NOW LIVE!
pip install Geology

# Verify installation
python -c "import geology; print('Geology package successfully installed!')"
```

# Launch interactive notebook
jupyter notebook demo.ipynb
```

#### Option 2: PyPI with Enhanced Features
```bash
# Install with visualization enhancements
pip install Geology[visualization]

# Install with geospatial capabilities  
pip install Geology[geospatial]

# Install with development tools
pip install Geology[dev]

# Install complete toolkit with all features
pip install Geology[all]
```

#### Option 3: Alternative Installation from Test PyPI (Legacy)
```bash
# Install from Test PyPI (development version)
pip install --index-url https://test.pypi.org/simple/ geology-Marios-toolkit

# Note: Use production PyPI version (Option 1) for stable releases
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

## 🎉 Package Release Status

### ✅ **OFFICIALLY RELEASED ON PYPI** 🚀:
- ✅ Package configuration and build system
- ✅ Distribution files created and validated  
- ✅ Successfully uploaded to Production PyPI
- ✅ Package name: **`Geology`**
- ✅ Global availability through `pip install Geology`

### 🌐 **Production Package Access**:

**Main PyPI (LIVE NOW):**
```bash
pip install Geology
```

### 📝 **Official Package Information**:
- **Package Name**: `Geology`
- **Version**: 1.0.0
- **PyPI URL**: https://pypi.org/project/Geology/
- **Package Size**: ~16-18 MB (includes comprehensive geological datasets)
- **Dependencies**: Complete scientific Python ecosystem
- **Release Date**: October 2025

---

**🌍 Advancing Geological Understanding Through Machine Learning 🌍**

[![Made with ❤️ for Geoscience](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20for-Geoscience-blue)](https://github.com/mariosgeo/Geology)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyPI](https://img.shields.io/badge/PyPI-Live%20Package-brightgreen)](https://pypi.org/project/Geology/)
[![VTK](https://img.shields.io/badge/VTK-3D%20Visualization-green)](https://vtk.org)
[![Open Science](https://img.shields.io/badge/Open-Science-orange)](https://github.com/mariosgeo/Geology)

**🎉 NOW LIVE ON PYPI:** `pip install Geology` 🌟