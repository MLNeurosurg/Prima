# End-to-end Inference Pipeline

This folder contains code for an end-to-end inference pipeline for applying Prima directly on a raw MRI study. The pipeline is designed to be user-friendly and easy to use and can be run on a local machine or a server. It will load the study, minimally process the data, and then pass it forward to the Prima model to generate the predicted radiologic diagnoses, referral, and prioritization recommendations.

The input to the pipeline is a folder containing raw MRI scans in DICOM format. The output is a JSON file.

Expected raw mri study folder structure:
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
Expected output JSON file structure:
```
{
    "diagnosis": "...",
    "referral": "...",
    "priority": "..."
}
```

