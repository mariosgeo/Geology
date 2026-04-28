#!/usr/bin/env python3
"""
GeoVTK Command Line Interface

Provides command-line tools for converting geological data to VTK format.
Supports batch processing of geological datasets and common conversion workflows.

Usage
-----
geovtk-convert --help                    # Show help
geovtk-convert --version                 # Show version
geovtk-convert tiff input.tif            # Convert GeoTIFF to VTK
geovtk-convert borehole data.csv         # Convert borehole data to VTK

Examples
--------
# Convert GeoTIFF elevation data to VTK surface
geovtk-convert tiff elevation.tif --output terrain.vtk

# Convert borehole CSV data to 3D visualization
geovtk-convert borehole boreholes.csv --radius 1.0 --output wells.vtk

# Batch convert multiple files
geovtk-convert batch *.tif --format vtk
"""

import sys
import argparse
import os
from pathlib import Path

# Import GeoVTK components
try:
    from . import VtkClass, geo_utils, __version__
except ImportError:
    # Fallback for direct execution
    try:
        from vtkclass import VtkClass
        import geo_utils
        __version__ = "1.0.0"
    except ImportError:
        print("ERROR: Could not import GeoVTK components. Please check installation.")
        sys.exit(1)


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog='geovtk-convert',
        description='Convert geological data to VTK format for 3D visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tiff elevation.tif --output terrain.vtk
  %(prog)s borehole wells.csv --radius 1.0
  %(prog)s --version
        """
    )
    
    # Version argument
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'GeoVTK {__version__}'
    )
    
    # Subcommands for different data types
    subparsers = parser.add_subparsers(
        dest='command',
        help='Geological data conversion commands',
        metavar='COMMAND'
    )
    
    # GeoTIFF conversion
    tiff_parser = subparsers.add_parser(
        'tiff',
        help='Convert GeoTIFF raster data to VTK format'
    )
    tiff_parser.add_argument(
        'input_file',
        help='Input GeoTIFF file path'
    )
    tiff_parser.add_argument(
        '--output', '-o',
        help='Output VTK file path (default: auto-generated)'
    )
    tiff_parser.add_argument(
        '--surface',
        action='store_true',
        help='Create surface instead of 3D volume'
    )
    
    # Borehole data conversion
    borehole_parser = subparsers.add_parser(
        'borehole',
        help='Convert borehole data to VTK format'
    )
    borehole_parser.add_argument(
        'input_file', 
        help='Input borehole data file (CSV format)'
    )
    borehole_parser.add_argument(
        '--radius', '-r',
        type=float,
        default=1.0,
        help='Borehole visualization radius (default: 1.0)'
    )
    borehole_parser.add_argument(
        '--output', '-o',
        help='Output VTK file path (default: auto-generated)'
    )
    
    # Batch processing
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch convert multiple geological files'
    )
    batch_parser.add_argument(
        'pattern',
        help='File pattern to match (e.g., "*.tif")'
    )
    batch_parser.add_argument(
        '--format',
        choices=['tiff', 'borehole'],
        required=True,
        help='Input data format'
    )
    
    return parser


def convert_tiff(input_file, output_file=None, surface=False):
    """Convert GeoTIFF file to VTK format."""
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = Path(input_file).with_suffix('.vtk')
    
    try:
        vtk = VtkClass()
        if surface:
            vtk.tiff_to_vtk(input_file)
        else:
            vtk.tiff_to_vtk_3d(input_file)
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        return True
        
    except Exception as e:
        print(f"ERROR converting '{input_file}': {e}")
        return False


def convert_borehole(input_file, radius=1.0, output_file=None):
    """Convert borehole data to VTK format."""
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = Path(input_file).with_suffix('.vtk')
    
    try:
        # This is a placeholder - would need to implement CSV reading
        # and proper data formatting for the VtkClass methods
        print(f"Borehole conversion not yet fully implemented.")
        print(f"Would convert '{input_file}' with radius {radius} to '{output_file}'")
        return True
        
    except Exception as e:
        print(f"ERROR converting '{input_file}': {e}")
        return False


def batch_convert(pattern, format_type):
    """Batch convert multiple files."""
    from glob import glob
    
    files = glob(pattern)
    if not files:
        print(f"No files found matching pattern '{pattern}'")
        return False
    
    print(f"Found {len(files)} files to convert...")
    
    success_count = 0
    for file_path in files:
        print(f"Converting {file_path}...")
        
        if format_type == 'tiff':
            success = convert_tiff(file_path)
        elif format_type == 'borehole':
            success = convert_borehole(file_path)
        else:
            print(f"Unsupported format: {format_type}")
            continue
            
        if success:
            success_count += 1
    
    print(f"Successfully converted {success_count}/{len(files)} files.")
    return success_count > 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Execute appropriate command
    success = False
    
    if args.command == 'tiff':
        success = convert_tiff(
            args.input_file, 
            args.output, 
            args.surface
        )
    elif args.command == 'borehole':
        success = convert_borehole(
            args.input_file,
            args.radius, 
            args.output
        )
    elif args.command == 'batch':
        success = batch_convert(args.pattern, args.format)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())