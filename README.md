# Geology - Biharmonic Inpainting for Geological Models

This repository contains a Python implementation for geological model reconstruction using inpainting techniques. The project focuses on filling gaps in geological data using biharmonic inpainting and one-vs-all classification methods.




## 🎯 Overview

This project demonstrates how to reconstruct geological subsurface models from sparse borehole data using advanced inpainting algorithms. The methodology combines:

- **Biharmonic Inpainting**: For smooth interpolation of geological boundaries
- **One-vs-All Classification**: For multi-class geological unit prediction
- **Weighted Interpolation**: For anisotropic geological features
- **Uncertainty Quantification**: To assess prediction confidence

## 📁 Project Structure

```
Geology/
├── demo.ipynb              # Main demonstration notebook
├── requirements.txt        # Python dependencies
├── geotop.npy             # 3D geological model data
├── top_layer.gpkg         # Geological layer data (GeoPackage)
├── gridder/               # Custom gridding utilities
├── geo_vtk/               # VTK-based geological visualization tools
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone this repository:
```bash
git clone https://github.com/mariosgeo/Geology.git
cd Geology
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook demo.ipynb
```

## 📊 Features

### 2D Geological Profile Reconstruction

The notebook demonstrates several geological scenarios:
- **Horizontal Layering**: Simple stratified deposits
- **Dipping Layers**: Inclined geological formations  
- **Fault Systems**: Discontinuous geological structures
- **Layer Pinch-outs**: Thinning and thickening sequences

### 3D Geological Model Inpainting

- Full 3D geological volume reconstruction
- Sparse borehole data interpolation
- Multi-scale geological feature preservation

### Validation and Uncertainty Analysis

- Cross-validation with held-out boreholes
- Confusion matrix analysis for classification accuracy
- Uncertainty quantification for prediction confidence
- Visual comparison of predicted vs. actual geology

## 🔬 Methodology

### Core Algorithm: One-vs-All Inpainting

```python
def one_vs_all(image, x_weight=1.0, y_weight=3.0):
    """
    Perform multi-class geological inpainting using one-vs-all approach
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input geological model with NaN values to inpaint
    x_weight : float
        Horizontal interpolation weight
    y_weight : float  
        Vertical interpolation weight (typically higher for geology)
    
    Returns:
    --------
    reconstructed_model : numpy.ndarray
        Inpainted geological model
    uncertainty : numpy.ndarray
        Prediction uncertainty percentage
    """
```

### Weighted Biharmonic Inpainting

The method uses anisotropic weights to account for geological principles:
- **Higher vertical weight**: Reflects geological layering
- **Lower horizontal weight**: Allows lateral geological variation

## 📈 Results

The methodology successfully reconstructs geological models from sparse borehole data with high accuracy. The following figures demonstrate key results:

### Figure 4: Geological Model Reconstruction
![Geological Model Reconstruction](images/figure_4.svg)

*This figure shows the step-by-step process of geological model reconstruction using biharmonic inpainting. The method effectively interpolates between sparse borehole data to create continuous geological boundaries while preserving structural features.*

### Figure 6: Validation and Uncertainty Analysis  
![Validation and Uncertainty Analysis](images/figure_6.svg)

*This figure demonstrates the validation approach using held-out boreholes and uncertainty quantification. The confusion matrix shows high classification accuracy, while uncertainty maps highlight areas where predictions are less confident.*

### Output Files

The notebook generates several outputs:

1. **Reconstructed Models**: PDF visualizations of inpainted geological cross-sections
2. **Validation Plots**: Comparison between predicted and actual geology  
3. **Confusion Matrices**: Classification accuracy assessment
4. **Uncertainty Maps**: Confidence levels of predictions
5. **3D VTK Models**: For visualization in ParaView or similar tools

## 🗂️ Data Files

### Essential Data Files

The repository includes several essential data files required for running the demonstrations:

| File | Description | Size | Purpose |
|------|-------------|------|---------|
| `geotop.npy` | 3D geological model data | ~Several MB | Main 3D geological volume for inpainting |
| `top_layer.gpkg` | Geological layer data | Variable | GeoPackage with borehole and geological information |

### Data Loading in Code

The notebook automatically loads these files:

```python
# Load 3D geological model
geotop = np.load('geotop.npy')

# Load geological layer data  
import geopandas as gpd
data = gpd.read_file('top_layer.gpkg')
```

**Note**: These data files are essential for reproducing the results and must be present in the repository root directory.

### Geological Classifications

The model uses a standardized geological classification system:

| Code | Geological Unit | Color |
|------|----------------|-------|
| 0    | Anthropogenic  | Grey  |
| 1    | Peat          | Brown |
| 2    | Clay          | Green |
| 3    | Silty Clay    | Light Green |
| 4    | Nothing       | White |
| 5    | Fine Sand     | Yellow |
| 6    | Medium Sand   | Gold |
| 7    | Coarse Sand   | Orange |
| 8    | Gravel        | Dark Orange |
| 9    | Shells        | Blue |

### Input Data Format

- **Borehole Data**: XYZ coordinates with geological classifications
- **Grid Spacing**: Configurable resolution for interpolation
- **Topographic Constraints**: Surface elevation boundaries

## 🛠️ Customization

### Adjusting Interpolation Weights

Modify the `x_weight` and `y_weight` parameters to control:
- **Geological continuity**: Higher weights = smoother interpolation
- **Structural complexity**: Lower weights = more geological detail

### Adding New Geological Units

Update the `labels` array and colormap to include additional geological classifications.

## 📚 Dependencies

Key packages used in this project:

- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization  
- **SciPy**: Scientific computing and interpolation
- **Scikit-image**: Image processing and inpainting algorithms
- **Scikit-learn**: Machine learning metrics and validation
- **GeoPandas**: Geospatial data handling
- **Shapely**: Geometric operations
- **VTK**: 3D visualization (via geo_vtk module)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

**Marios Karaoulis**  
- GitHub: [@mariosgeo](https://github.com/mariosgeo)

## 🙏 Acknowledgments

- This work builds upon established geological modeling principles
- Inpainting algorithms adapted from scikit-image library
- Geological data classification follows industry standards

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{karaoulis2025geology,
  author = {Karaoulis, Marios},
  title = {Geology: Machine Learning Inpainting for Geological Models},
  year = {2025},
  url = {https://github.com/mariosgeo/Geology}
}
```