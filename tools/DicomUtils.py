import os
import logging
import natsort
import SimpleITK as sitk
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class DicomUtils:
    """Utility class for handling DICOM files and series."""

    def __init__(self, study_dir: Optional[str] = None):
        """
        Initialize DicomUtils.
        
        Args:
            study_dir: Optional path to study directory
        """
        self.study_dir = study_dir
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def print_sitk_info(image: sitk.Image, type_: str = "Original", return_dict: bool = False) -> Optional[Dict]:
        """
        Print or return SimpleITK image information.
        
        Args:
            image: SimpleITK image
            type_: Type of image for logging
            return_dict: Whether to return info as dictionary
            
        Returns:
            Dictionary of image info if return_dict is True, None otherwise
        """
        orient = sitk.DICOMOrientImageFilter()

        info_dict = {
            f'{type_}Size': image.GetSize(),
            f'{type_}Origin': image.GetOrigin(),
            f'{type_}Spacing': image.GetSpacing(),
            f'{type_}Direction': image.GetDirection(),
            f'{type_}Orientation': orient.GetOrientationFromDirectionCosines(image.GetDirection()),
            f'{type_}Pixel_type': image.GetPixelIDTypeAsString()
        }

        for k, v in info_dict.items():
            logging.info(f"{k}: {v}")

        return info_dict if return_dict else None

    @staticmethod   
    def subsample_series(image: sitk.Image, target_slices: int = 500) -> sitk.Image:
        """
        Subsample a series to a target number of slices.
        
        Args:
            image: Input SimpleITK image
            target_slices: Target number of slices
            
        Returns:
            Subsampled SimpleITK image
        """
        try:
            size = list(image.GetSize())
            original_slices = size[2]

            step = original_slices / float(target_slices)
            indices = [int(i * step) for i in range(target_slices)]

            extractor = sitk.ExtractImageFilter()
            subsampled_slices = []
            for idx in indices:
                extractor.SetSize([size[0], size[1], 0])
                extractor.SetIndex([0, 0, idx])
                subsampled_slice = extractor.Execute(image)
                subsampled_slices.append(subsampled_slice)

            return sitk.JoinSeries(subsampled_slices)
        except Exception as e:
            raise RuntimeError(f"Failed to subsample series: {str(e)}")

    @staticmethod
    def filter_dicom_series(file_paths: List[str]) -> List[str]:
        """
        Filter DICOM series to get the most common size.
        
        Args:
            file_paths: List of DICOM file paths
            
        Returns:
            List of filtered file paths
        """
        try:
            # Dictionary to store file paths keyed by their size
            size_to_paths = {}
            for file_path in file_paths:
                reader = sitk.ImageFileReader()
                reader.SetFileName(file_path)
                reader.ReadImageInformation()
                size = reader.GetSize()
                if size not in size_to_paths:
                    size_to_paths[size] = []
                size_to_paths[size].append(file_path)

            common_size_files = max(size_to_paths.values(), key=len)
            return common_size_files
        except Exception as e:
            raise RuntimeError(f"Failed to filter DICOM series: {str(e)}")

    @staticmethod
    def read_dicom_series(
        directory: str,
        new_orientation: Optional[str] = None,
        new_size: Tuple[Optional[int], Optional[int], Optional[int]] = (256, 256, None),
        save_path: Optional[str] = None,
        len_threshold: int = 500
    ) -> Tuple[sitk.Image, List[str], Dict]:
        """
        Read and process a DICOM series.
        
        Args:
            directory: Path to DICOM series directory
            new_orientation: Optional new orientation
            new_size: Optional new size (x, y, z)
            save_path: Optional path to save processed image
            len_threshold: Threshold for series length
            
        Returns:
            Tuple of (processed image, DICOM file names, info dictionary)
        """
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(directory)
            dicom_names = natsort.natsorted(DicomUtils.filter_dicom_series(dicom_names))
            reader.SetFileNames(dicom_names)
            logging.info('*' * 10)
            
            # Execute dicom reader
            image = reader.Execute()
            info_dict = {}
            original_dict = DicomUtils.print_sitk_info(image, "Original", return_dict=True)
            info_dict.update(original_dict)

            # Resize the image if new_size is specified
            if new_size[0] is not None and new_size[1] is not None:
                original_size = image.GetSize()
                original_spacing = image.GetSpacing()

                # Calculate new spacing for X and Y, with Z spacing unchanged
                new_spacing = [
                    (original_size[0] * original_spacing[0]) / new_size[0],
                    (original_size[1] * original_spacing[1]) / new_size[1],
                    original_spacing[2] 
                ]
                if new_size[2] is None:
                    new_size = (new_size[0], new_size[1], original_size[2])

                # Create the reference image with new size and spacing
                reference_image = sitk.Image(new_size, image.GetPixelIDValue())
                reference_image.SetOrigin(image.GetOrigin())
                reference_image.SetDirection(image.GetDirection())
                reference_image.SetSpacing(new_spacing)

                # resample
                image = sitk.Resample(image, reference_image, sitk.Transform(),
                                    sitk.sitkLinear, image.GetPixelIDValue())

            logging.info('=' * 10)
            DicomUtils.print_sitk_info(image, type_="AfterResize")

            if new_orientation is not None:
                image = sitk.DICOMOrient(image, new_orientation)
                # Print info after orientation (if applied)
                logging.info('=' * 10)
                new_dict = DicomUtils.print_sitk_info(image, type_="Reorient", return_dict=True)
                info_dict.update(new_dict) 

            # Optionally save the image
            if save_path is not None:
                sitk.WriteImage(image, save_path)
                logging.info(f"Image saved to {save_path}")
                logging.info('*' * 10)
                
            return image, dicom_names, info_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM series: {str(e)}")

    @staticmethod
    def load_mri_study(study_dir: str) -> Tuple[List[sitk.Image], List[str]]:
        """
        Load all series from an MRI study directory.
        
        Args:
            study_dir: Path to study directory
            
        Returns:
            Tuple of (list of series images, list of series names)
        """
        try:
            logging.info('Loading MRI studies')
            series_list = natsort.natsorted(os.listdir(study_dir))
            mri_study = []
            
            for series in series_list:
                series_path = os.path.join(study_dir, series)
                series_image, _, _ = DicomUtils.read_dicom_series(series_path)
                mri_study.append(series_image)
                
            return mri_study, series_list
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MRI study: {str(e)}")


if __name__ == '__main__':
    print('Tools for dicom data')