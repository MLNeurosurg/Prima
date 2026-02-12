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

To run the end-to-end pipeline, first you need to download both the Prima model and head weights (link to be provided) and the VQ-VAE weights (link to be provided), then update `configs/pipeline_config.yaml` with and fill in `study_dir` (where you stored your study data), `output_dir` (where do you want the output json to be stored), `ckpt_dir` under `tokenizer_model_config` (or within `configs/sample_tokenizer_config.json`) to be where you stored the VQVAE checkpoint, and `full_model_ckpt` under `prima_model_config` (or within `configs/sample_prima_model_config.json`) to where you stored the Prima model and head weight pt file. Then, from the main repository directory, run
```
python /end-to-end_inference_pipeline/pipeline.py --config configs/pipeline_config.yaml
```

