# Under construction

# End-to-end Inference Pipeline

This folder contains code for an end-to-end inference pipeline for applying Prima directly on raw MRI scans. The pipeline is designed to be user-friendly and easy to use. The pipeline is designed to be run on a local machine or a server. The pipeline can be used to perform inference on raw, uncurated MRI studies.

The input to the pipeline is a folder containing raw MRI scans in DICOM format. The output is a CSV file containing the predicted radiologic diagnoses for each MRI study.

Expected input folder structure:
```
Study_dir/
    series1/
        image1.dcm
        image2.dcm
        ...
    series2/
        image1.dcm
        image2.dcm
        ...
    series3/
        image1.dcm
        image2.dcm
        ...
    series4/
        image1.dcm
        image2.dcm
        ...
    ...
```


