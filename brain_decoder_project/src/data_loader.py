"""
Data Loader for Stimulus and fMRI Data

This module provides comprehensive data loading functionality for neuroscience research,
supporting various formats for both stimulus and fMRI data.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Core scientific computing
try:
    import scipy.io as sio
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. MATLAB file loading will be limited.")

# Neuroimaging
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("Nibabel not available. NIfTI file loading will be limited.")

# Image processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    warnings.warn("PIL not available. Image loading will be limited.")

# Video processing
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    warnings.warn("OpenCV not available. Video loading will be limited.")


class DataLoader:
    """
    Comprehensive data loader for stimulus and fMRI data.

    Supports multiple file formats and provides preprocessing utilities.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader.

        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = {
            'matlab': ['.mat'],
            'nifti': ['.nii', '.nii.gz'],
            'image': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'],
            'video': ['.mp4', '.avi', '.mov', '.mkv'],
            'text': ['.txt', '.csv', '.tsv'],
            'numpy': ['.npy', '.npz']
        }

    def load_data(self, filename: str, data_type: str = 'auto') -> Dict[str, Any]:
        """
        Load data from file with automatic format detection.

        Args:
            filename (str): Name of the file to load
            data_type (str): Type of data ('auto', 'stimulus', 'fmri', 'matlab', etc.)

        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect format if not specified
        if data_type == 'auto':
            data_type = self._detect_format(filepath)

        # Load based on detected/specified format
        if data_type == 'matlab':
            return self.load_matlab(filepath)
        elif data_type == 'nifti':
            return self.load_nifti(filepath)
        elif data_type == 'image':
            return self.load_image(filepath)
        elif data_type == 'video':
            return self.load_video(filepath)
        elif data_type == 'text':
            return self.load_text(filepath)
        elif data_type == 'numpy':
            return self.load_numpy(filepath)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def _detect_format(self, filepath: Path) -> str:
        """Detect file format based on extension."""
        suffix = filepath.suffix.lower()

        for format_type, extensions in self.supported_formats.items():
            if suffix in extensions:
                return format_type

        # Special case for .gz files
        if suffix == '.gz' and filepath.stem.endswith('.nii'):
            return 'nifti'

        raise ValueError(f"Unsupported file format: {suffix}")

    def load_matlab(self, filepath: Path) -> Dict[str, Any]:
        """
        Load MATLAB .mat files.

        Args:
            filepath (Path): Path to the .mat file

        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy is required for loading MATLAB files")

        try:
            # Load the .mat file
            mat_data = sio.loadmat(str(filepath))

            # Remove MATLAB metadata
            clean_data = {k: v for k, v in mat_data.items()
                         if not k.startswith('__')}

            # Extract main data arrays
            data_arrays = {}
            metadata = {
                'filename': filepath.name,
                'format': 'matlab',
                'variables': list(clean_data.keys())
            }

            for key, value in clean_data.items():
                if isinstance(value, np.ndarray):
                    data_arrays[key] = value
                    metadata[f'{key}_shape'] = value.shape
                    metadata[f'{key}_dtype'] = str(value.dtype)

            return {
                'data': data_arrays,
                'metadata': metadata,
                'raw': clean_data
            }

        except Exception as e:
            raise RuntimeError(f"Error loading MATLAB file {filepath}: {str(e)}")

    def load_nifti(self, filepath: Path) -> Dict[str, Any]:
        """
        Load NIfTI files (common fMRI format).

        Args:
            filepath (Path): Path to the NIfTI file

        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        if not HAS_NIBABEL:
            raise ImportError("Nibabel is required for loading NIfTI files")

        try:
            # Load NIfTI file
            nii_img = nib.load(str(filepath))
            data = nii_img.get_fdata()

            # Extract metadata
            header = nii_img.header
            affine = nii_img.affine

            metadata = {
                'filename': filepath.name,
                'format': 'nifti',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'voxel_size': header.get_zooms(),
                'data_type': header.get_data_dtype(),
                'affine_matrix': affine,
                'header': dict(header)
            }

            return {
                'data': data,
                'metadata': metadata,
                'affine': affine,
                'header': header
            }

        except Exception as e:
            raise RuntimeError(f"Error loading NIfTI file {filepath}: {str(e)}")

    def load_image(self, filepath: Path) -> Dict[str, Any]:
        """
        Load image files (stimulus data).

        Args:
            filepath (Path): Path to the image file

        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        if not HAS_PIL:
            raise ImportError("PIL is required for loading image files")

        try:
            # Load image
            img = Image.open(filepath)

            # Convert to numpy array
            img_array = np.array(img)

            metadata = {
                'filename': filepath.name,
                'format': 'image',
                'shape': img_array.shape,
                'dtype': str(img_array.dtype),
                'mode': img.mode,
                'size': img.size
            }

            return {
                'data': img_array,
                'metadata': metadata,
                'pil_image': img
            }

        except Exception as e:
            raise RuntimeError(f"Error loading image file {filepath}: {str(e)}")

    def load_video(self, filepath: Path) -> Dict[str, Any]:
        """
        Load video files (stimulus data).

        Args:
            filepath (Path): Path to the video file

        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for loading video files")

        try:
            # Open video
            cap = cv2.VideoCapture(str(filepath))

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            # Convert to numpy array
            video_array = np.array(frames)

            metadata = {
                'filename': filepath.name,
                'format': 'video',
                'shape': video_array.shape,
                'dtype': str(video_array.dtype),
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': frame_count / fps if fps > 0 else 0
            }

            return {
                'data': video_array,
                'metadata': metadata
            }

        except Exception as e:
            raise RuntimeError(f"Error loading video file {filepath}: {str(e)}")

    def load_text(self, filepath: Path) -> Dict[str, Any]:
        """
        Load text files (CSV, TSV, TXT).

        Args:
            filepath (Path): Path to the text file

        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        try:
            suffix = filepath.suffix.lower()

            if suffix == '.csv':
                data = pd.read_csv(filepath)
            elif suffix == '.tsv':
                data = pd.read_csv(filepath, sep='\t')
            else:  # .txt or other
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                data = content

            metadata = {
                'filename': filepath.name,
                'format': 'text',
                'file_type': suffix,
                'size_bytes': filepath.stat().st_size
            }

            if isinstance(data, pd.DataFrame):
                metadata.update({
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
                })

            return {
                'data': data,
                'metadata': metadata
            }

        except Exception as e:
            raise RuntimeError(f"Error loading text file {filepath}: {str(e)}")

    def load_numpy(self, filepath: Path) -> Dict[str, Any]:
        """
        Load NumPy files (.npy, .npz).

        Args:
            filepath (Path): Path to the NumPy file

        Returns:
            Dict[str, Any]: Dictionary containing loaded data and metadata
        """
        try:
            suffix = filepath.suffix.lower()

            if suffix == '.npy':
                data = np.load(filepath)
                metadata = {
                    'filename': filepath.name,
                    'format': 'numpy',
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                }
                return {
                    'data': data,
                    'metadata': metadata
                }

            elif suffix == '.npz':
                data = np.load(filepath)
                arrays = {key: data[key] for key in data.files}
                metadata = {
                    'filename': filepath.name,
                    'format': 'numpy_compressed',
                    'arrays': list(data.files)
                }

                for key, array in arrays.items():
                    metadata[f'{key}_shape'] = array.shape
                    metadata[f'{key}_dtype'] = str(array.dtype)

                return {
                    'data': arrays,
                    'metadata': metadata,
                    'raw': data
                }

        except Exception as e:
            raise RuntimeError(f"Error loading NumPy file {filepath}: {str(e)}")

    def list_files(self, pattern: str = "*") -> List[Path]:
        """
        List files in the data directory matching a pattern.

        Args:
            pattern (str): Glob pattern to match files

        Returns:
            List[Path]: List of matching file paths
        """
        return list(self.data_dir.glob(pattern))

    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get basic information about a file without loading it.

        Args:
            filename (str): Name of the file

        Returns:
            Dict[str, Any]: File information
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        stat = filepath.stat()

        info = {
            'filename': filepath.name,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime,
            'extension': filepath.suffix.lower(),
            'detected_format': None
        }

        try:
            info['detected_format'] = self._detect_format(filepath)
        except ValueError:
            info['detected_format'] = 'unknown'

        return info

    def load_stimulus_batch(self, pattern: str = "*.png") -> Dict[str, Any]:
        """
        Load multiple stimulus files matching a pattern.

        Args:
            pattern (str): Glob pattern to match stimulus files

        Returns:
            Dict[str, Any]: Dictionary containing batch data and metadata
        """
        files = self.list_files(pattern)

        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")

        batch_data = []
        metadata_list = []

        for filepath in sorted(files):
            try:
                result = self.load_data(filepath.name)
                batch_data.append(result['data'])
                metadata_list.append(result['metadata'])
            except Exception as e:
                warnings.warn(f"Failed to load {filepath.name}: {str(e)}")
                continue

        if not batch_data:
            raise RuntimeError("No files could be loaded successfully")

        # Stack data if possible
        try:
            stacked_data = np.stack(batch_data)
        except ValueError:
            # If stacking fails, keep as list
            stacked_data = batch_data

        batch_metadata = {
            'num_files': len(batch_data),
            'pattern': pattern,
            'files': [meta['filename'] for meta in metadata_list],
            'individual_metadata': metadata_list
        }

        if isinstance(stacked_data, np.ndarray):
            batch_metadata.update({
                'batch_shape': stacked_data.shape,
                'batch_dtype': str(stacked_data.dtype)
            })

        return {
            'data': stacked_data,
            'metadata': batch_metadata
        }

    def preprocess_fmri(self, data: np.ndarray,
                       normalize: bool = True,
                       detrend: bool = True,
                       standardize: bool = True) -> np.ndarray:
        """
        Basic preprocessing for fMRI data.

        Args:
            data (np.ndarray): fMRI data (typically 4D: x, y, z, time)
            normalize (bool): Whether to normalize to [0, 1]
            detrend (bool): Whether to remove linear trend
            standardize (bool): Whether to standardize (z-score)

        Returns:
            np.ndarray: Preprocessed data
        """
        processed_data = data.copy()

        # Handle different dimensionalities
        if processed_data.ndim == 4:
            # 4D data: spatial + time
            time_axis = -1
        elif processed_data.ndim == 2:
            # 2D data: voxels x time
            time_axis = -1
        else:
            warnings.warn(f"Unexpected data dimensionality: {processed_data.ndim}")
            time_axis = -1

        # Normalize to [0, 1]
        if normalize:
            data_min = np.min(processed_data)
            data_max = np.max(processed_data)
            if data_max > data_min:
                processed_data = (processed_data - data_min) / (data_max - data_min)

        # Detrend (remove linear trend along time axis)
        if detrend and HAS_SCIPY:
            from scipy import signal
            processed_data = signal.detrend(processed_data, axis=time_axis)

        # Standardize (z-score along time axis)
        if standardize:
            mean = np.mean(processed_data, axis=time_axis, keepdims=True)
            std = np.std(processed_data, axis=time_axis, keepdims=True)
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            processed_data = (processed_data - mean) / std

        return processed_data

    def preprocess_stimulus(self, data: np.ndarray,
                          resize: Optional[Tuple[int, int]] = None,
                          normalize: bool = True,
                          grayscale: bool = False) -> np.ndarray:
        """
        Basic preprocessing for stimulus data (images/videos).

        Args:
            data (np.ndarray): Stimulus data
            resize (Optional[Tuple[int, int]]): Target size (height, width)
            normalize (bool): Whether to normalize to [0, 1]
            grayscale (bool): Whether to convert to grayscale

        Returns:
            np.ndarray: Preprocessed data
        """
        processed_data = data.copy()

        # Convert to grayscale if requested
        if grayscale and processed_data.ndim >= 3:
            if processed_data.shape[-1] == 3:  # RGB
                # Use standard RGB to grayscale conversion
                weights = np.array([0.299, 0.587, 0.114])
                processed_data = np.dot(processed_data, weights)
            elif processed_data.shape[-1] == 4:  # RGBA
                # Convert RGBA to RGB first, then grayscale
                rgb_data = processed_data[..., :3]
                weights = np.array([0.299, 0.587, 0.114])
                processed_data = np.dot(rgb_data, weights)

        # Resize if requested
        if resize is not None and HAS_SCIPY:
            target_height, target_width = resize

            if processed_data.ndim == 2:  # Single grayscale image
                zoom_factors = (target_height / processed_data.shape[0],
                              target_width / processed_data.shape[1])
                processed_data = zoom(processed_data, zoom_factors)

            elif processed_data.ndim == 3:  # Single color image or batch of grayscale
                if processed_data.shape[-1] in [1, 3, 4]:  # Color image
                    zoom_factors = (target_height / processed_data.shape[0],
                                  target_width / processed_data.shape[1], 1)
                else:  # Batch of grayscale images
                    zoom_factors = (1, target_height / processed_data.shape[1],
                                  target_width / processed_data.shape[2])
                processed_data = zoom(processed_data, zoom_factors)

            elif processed_data.ndim == 4:  # Batch of color images or video
                zoom_factors = (1, target_height / processed_data.shape[1],
                              target_width / processed_data.shape[2], 1)
                processed_data = zoom(processed_data, zoom_factors)

        # Normalize to [0, 1]
        if normalize:
            if processed_data.dtype == np.uint8:
                processed_data = processed_data.astype(np.float32) / 255.0
            else:
                data_min = np.min(processed_data)
                data_max = np.max(processed_data)
                if data_max > data_min:
                    processed_data = (processed_data - data_min) / (data_max - data_min)

        return processed_data


# Convenience functions for quick data loading
def load_stimulus_data(data_dir: str = "data", filename: str = None,
                      pattern: str = None, preprocess: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load stimulus data.

    Args:
        data_dir (str): Path to data directory
        filename (str): Specific file to load (if None, uses pattern)
        pattern (str): Pattern to match multiple files (if None, uses filename)
        preprocess (bool): Whether to apply basic preprocessing

    Returns:
        Dict[str, Any]: Loaded data and metadata
    """
    loader = DataLoader(data_dir)

    if filename is not None:
        result = loader.load_data(filename)
        if preprocess and isinstance(result['data'], np.ndarray):
            result['data'] = loader.preprocess_stimulus(result['data'])
        return result

    elif pattern is not None:
        result = loader.load_stimulus_batch(pattern)
        if preprocess and isinstance(result['data'], np.ndarray):
            result['data'] = loader.preprocess_stimulus(result['data'])
        return result

    else:
        raise ValueError("Either filename or pattern must be specified")


def load_fmri_data(filename: str, data_dir: str = "data",
                  preprocess: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load fMRI data.

    Args:
        data_dir (str): Path to data directory
        filename (str): Name of the fMRI file
        preprocess (bool): Whether to apply basic preprocessing

    Returns:
        Dict[str, Any]: Loaded data and metadata
    """
    loader = DataLoader(data_dir)
    result = loader.load_data(filename)

    if preprocess and isinstance(result['data'], np.ndarray):
        result['data'] = loader.preprocess_fmri(result['data'])

    return result


def load_matlab_data(filename: str, data_dir: str = "data") -> Dict[str, Any]:
    """
    Convenience function to load MATLAB data.

    Args:
        data_dir (str): Path to data directory
        filename (str): Name of the MATLAB file

    Returns:
        Dict[str, Any]: Loaded data and metadata
    """
    loader = DataLoader(data_dir)
    return loader.load_data(filename, data_type='matlab')


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Data Loader Example Usage")
    print("=" * 40)

    # Initialize loader
    loader = DataLoader("data")

    # List available files
    print("\nAvailable files:")
    files = loader.list_files()
    for file in files:
        info = loader.get_file_info(file.name)
        print(f"  {file.name} ({info['detected_format']}, {info['size_mb']:.2f} MB)")

    # Load the existing MATLAB file
    if files:
        try:
            # Try to load the first file
            first_file = files[0].name
            print(f"\nLoading {first_file}...")

            result = loader.load_data(first_file)
            print(f"Successfully loaded {first_file}")
            print(f"Format: {result['metadata']['format']}")

            if 'data' in result:
                if isinstance(result['data'], dict):
                    print("Data variables:")
                    for key, value in result['data'].items():
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"  {key}: {type(value)}")
                elif isinstance(result['data'], np.ndarray):
                    print(f"Data shape: {result['data'].shape}")
                    print(f"Data type: {result['data'].dtype}")

        except Exception as e:
            print(f"Error loading file: {str(e)}")

    print("\nData loader ready for use!")
    print("\nExample usage:")
    print("  loader = DataLoader('data')")
    print("  result = loader.load_data('your_file.mat')")
    print("  data = result['data']")
    print("  metadata = result['metadata']")