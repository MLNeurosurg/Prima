import os
import logging
import natsort
import SimpleITK as sitk

from pathlib import Path


class DicomUtils:
    def __init__(self, study_dir):
        self.study_dir = study_dir

    def print_sitk_info(image, type_="Original", return_dict=False):
        orient = sitk.DICOMOrientImageFilter()

        info_dict = {}
        info_dict[f'{type_}Size'] = image.GetSize()
        info_dict[f'{type_}Origin'] = image.GetOrigin()
        info_dict[f'{type_}Spacing'] = image.GetSpacing()
        info_dict[f'{type_}Direction'] = image.GetDirection()
        info_dict[
            f'{type_}Orientation'] = orient.GetOrientationFromDirectionCosines(
                image.GetDirection())
        info_dict[f'{type_}Pixel_type'] = image.GetPixelIDTypeAsString()

        # log k,v from the dict
        for k, v in info_dict.items():
            logging.info(f"{k}: {v}")

        if return_dict:
            return info_dict

    def subsample_series(image, target_slices=500):
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

    def filter_dicom_series(file_paths):
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

    def read_dicom_series(self, directory,
                      new_orientation=None,
                      new_size=(256, 256, None),
                      save_path=None,
                      len_threshold=500):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        dicom_names = natsort.natsorted(self.filter_dicom_series(dicom_names))
        reader.SetFileNames(dicom_names)
        logging.info('*' * 10)
        # Execute dicom reader
        image = reader.Execute()
        info_dict = {}
        original_dict = self.print_sitk_info(image, "Original", return_dict=True)

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
                new_size[2] = original_size[2]

            # Create the reference image with new size and spacing
            reference_image = sitk.Image(new_size, image.GetPixelIDValue())
            reference_image.SetOrigin(image.GetOrigin())
            reference_image.SetDirection(image.GetDirection())
            reference_image.SetSpacing(new_spacing)

            # resample
            image = sitk.Resample(image, reference_image, sitk.Transform(),
                                sitk.sitkLinear, image.GetPixelIDValue())

        logging.info('=' * 10)
        self.print_sitk_info(image, type_="AfterResize")

        if new_orientation is not None:
            image = sitk.DICOMOrient(image, new_orientation)
            # Print info after orientation (if applied)
            logging.info('=' * 10)
            new_dict = self.print_sitk_info(image, type_="Reorient", return_dict=True)
        info_dict.update(new_dict) 

        # Optionally save the image
        if save_path is not None:
            sitk.WriteImage(image, save_path)
            logging.info(f"Image saved to {save_path}")
            logging.info('*' * 10)
        return image, dicom_names, info_dict

    def load_mri_study(self):
        '''
        Load the MRI study from the study directory
        '''
        logging.info('Loading MRI studies')
        series_list = natsort.natsorted(os.listdir(self.study_dir))
        mri_study = []
        for series in series_list:
            series_path = os.path.join(self.study_dir, series)
            series_image, _, _ = self.read_dicom_series(series_path)
            mri_study.append(series_image)
        return mri_study



if __name__ == '__main__':
    print('Tools for dicom data')