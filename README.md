

# Prima: General-Purpose Brain MRI Vision Language Model trained on Health System-scale Data 

<img src="https://github.com/user-attachments/assets/7028090e-dfbf-4358-b99c-3240a1918e87" width="300px" align="right" />


This is the official code repository for paper submission "[Learning neuroimaging models from health system-scale data](https://arxiv.org/abs/2509.18638v1)", where we introduced Prima, the first general-purpose MRI VLM trained on health system-scale data. [Demo Website](https://prima.mlins.org)

```
@misc{lyu2025learningneuroimagingmodelshealth,
      title={Learning neuroimaging models from health system-scale data}, 
      author={Yiwei Lyu and Samir Harake and Asadur Chowdury and Soumyanil Banerjee and Rachel Gologorsky and Shixuan Liu and Anna-Katharina Meissner and Akshay Rao and Chenhui Zhao and Akhil Kondepudi and Cheng Jiang and Xinhai Hou and Rushikesh S. Joshi and Volker Neuschmelting and Ashok Srinivasan and Dawn Kleindorfer and Brian Athey and Vikas Gulani and Aditya Pandey and Honglak Lee and Todd Hollon},
      year={2025},
      eprint={2509.18638},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.18638}, 
}
```

## Model Weights

Download the following weights and place them as indicated in the [end-to-end inference pipeline](end-to-end_inference_pipeline) config (see [configs](configs) for examples).

| Model | Download |
|-------|----------|
| **Full PRIMA model** (`primafullmodel107.pt`) | [![Download Full Model](https://img.shields.io/badge/Download-Full_Model_Weights-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/119kKMcdk1GPww69IQAf6JkXuNMIEEAIk/view) |
| **Tokenizer (VQ-VAE)** (`vqvae_model_step16799.pth`) | [![Download Tokenizer](https://img.shields.io/badge/Download-Tokenizer_Weights-34A853?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/11EitVfPVXmdPSJviQQ5ZKasFNbQqD5Bt/view?usp=drive_link) |

- **Full model:** use in your PRIMA config as `full_model_ckpt` (e.g. in `prima_config.json` or `sample_prima_config.json`).
- **Tokenizer:** place the checkpoint path in your tokenizer config’s `vqvae_config.ckpt_path` (e.g. in `config.json` under the tokenizer model directory).
- **Note:** The priority model within the full model ckpt has been corrected on 2/19/2026. For best performance, please redownload the full model checkpoint above if you downloaded it before 2/19/2026.

### Quick start: download weights and run a test

1. **Set up a virtual environment** and install dependencies from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download the model weights** (requires `gdown`, which is in `requirements.txt`):

```bash
python tools/download_models_and_setup_test.py
```

3. **Place a DICOM study** folder under `test/test_mri_case/` (e.g. `test/test_mri_case/MY_STUDY_ID/`), set `study_dir` in `configs/test_pipeline_config.yaml` to that path, then run the pipeline:

```bash
python end-to-end_inference_pipeline/pipeline.py --config configs/test_pipeline_config.yaml
```

Predictions will be written to `test/test_output/{study_id}_predictions.json` (e.g. `MY_STUDY_ID_predictions.json`).

## Overview

![1738022426126-795fc098-7897-47f6-941d-8724d4c5638a_1](https://github.com/user-attachments/assets/be841dcb-a446-4e00-b33d-17901c78557f)


The global demand for magnetic resonance imaging (MRI) studies has risen steadily, placing significant strain
on health systems, prolonging turnaround times, and intensifying physician burnout. These
challenges disproportionately impact patients in low-resource and rural settings. Here, we utilized
a large academic health system as a data engine to develop Prima, the first vision language model
(VLM) serving as an AI foundation for neuroimaging that supports real-world, clinical MRI studies
as input. We curated a health system-scale dataset, UM-220K, which consists of over 220,000 MRI studies (including over 5.6 million 3D MRI sequences and 362 million 2D MRI images) together with their corresponding radiology reports.
Trained on over UM-220K via contrastive objective, Prima uses a hierarchical vision architecture that
provides general and transferable MRI representation features. 


## Results

![Fig2](https://github.com/user-attachments/assets/1272a11d-b35b-4433-857d-2a741054f345)

Prima was tested in a 1-year, prospective, health
system-wide study that included 30K MRI studies. Across 52 radiologic diagnoses from the major
neurologic disorders, including neoplastic, inflammatory, infectious, and developmental lesions,
Prima achieved a mean diagnostic area under the ROC curve of 90.1 ± 5.0%  (**ab**), outperforming other state-of-the-art general and medical AI models by a large margin. We show that the scale of training data is essential in obtaining good performance (**c**). In addition, Prima offers preliminary
differential diagnoses, worklist priority for radiologists (**d**), and clinical referral recommendations (**e**) across
diverse patient demographics and MRI systems. Prima features are also transferable to tasks not usually included in clinical radiology reports, like Alzheimer's disease, autism and brain age prediction (**f**). 

## Explainability

![fig3](https://github.com/user-attachments/assets/3ad4fb2d-60d7-4fc0-b627-8d1b931906d9)

Prima's predictions are explainable through [LIME](https://arxiv.org/abs/1602.04938). Three clinical vignettes demonstrate Prima’s
explainability using Local Interpretable Model-Agnostic Explanations (LIME). The left panels show patient MRIs
at initial presentation (MRI Pre) with the top Prima logits and the top-3 volume tokens identified by LIME. The
center bar charts depict changes in Prima logits between the initial presentation (MRI Pre) and after progression
or intervention (MRI Post). The right panels display patient MRIs following their clinical courses (MRI Post). **a**,
Clinical vignette of a diffuse low-grade glioma patient, status post (s/p) subtotal resection, who experienced
tumor progression and malignant transformation seven years after treatment. Prima accurately identified new
regions of contrast enhancement, consistent with malignant glioma. **b**, Clinical vignette of a patient with a
spontaneous brain abscess who underwent surgical drainage and antibiotic treatment, resulting in resolution.
**c**, Clinical vignette of a pediatric patient with a history of myelomeningocele and shunted hydrocephalus.
At baseline, the patient had mild ventriculomegaly but presented with acute hydrocephalus following shunt
malfunction. Prima accurately predicted the worsening of ventriculomegaly. Interactive demonstration can be
found at [Demo Website](https://prima.mlins.org).

## Fairness

![Fig4 (1)](https://github.com/user-attachments/assets/73b65d74-1471-4808-8147-57f15451702f)

Prima demonstrates algorithmic fairness across sensitive groups and can help mitigate health system biases. There is inequality in the odds of long patient waiting time based on **a.** county population, **b.** location, and **c.** scheduling day of the week. Despite these biases, Prima maintained consistent fairness across all above situations.

## Prima Architecture

![Ext_Data_Fig1](https://github.com/user-attachments/assets/e6086ec9-78c4-4617-8639-5681b302eeba)

**Expanded Workflow and Prima Architecture a,** Our health system Sectra
PACS server was queried for all cranial MRIs. We then filtered these MRIs based on the availability of
an associated radiology report and having a minimum of 2 series per study. We then ensured that
all metadata was present, resulting in a total of 221,147 UM-220K dataset. **b,** Overview of the stages
of training Prima on UM-220K, which includes volume tokenization, hierarchical ViT training with
CLIP objective function, and transfer learning to predict radiologic diagnoses. An LLM provides
radiology report summarization and diagnostic labels for reliable, accurate, and scalable vision-
language modeling. **c,** Volume tokenization stage involves dividing each MRI volume into smaller
subvolume patches of shape 32x32x4, removing background tokens, and encoding each subvolume
using a VQ-VAE encoder. We provide code for preprocessing and VQ-VAE tokenization under [preprocessing and tokenization](preprocessing_and_tokenization).

The latent VQ-VAE tokens are then passed forward to the sequence ViT
with the concatenated positional encodings. **d,** The hierarchical ViT is trained using a CLIP objective
on frozen volume token features. The sequence ViT is a multimodal transformer that takes as input
both the volume tokens and the embedded free-text sequence description. The series registers are
passed forward to the study ViT that outputs a single representation for the full MRI study. The paired
reports are summarized and passed through a pre-trained neuroradiology model to align the MRI
study and the paired report. **e,** A transfer learning strategy is used such that the volume tokenizer,
sequence and study transformers are frozen, and an MLP is trained on the learned study features for
radiologic and clinical task prediction. We provide code for training the hierarchical vision transformer using CLIP objective, as well as code for training the task-specific MLPs and evaluating on prospective test set, under [Prima training and evaluation](Prima_training_and_evaluation).

We also provide an **end-to-end** inference pipeline for applying Prima directly on raw MRI study at [end-to-end inference pipeline](end-to-end_inference_pipeline).

# Repository Structure

This repository provides code and instructions for 3 parts of our project

(1) Under [preprocessing and tokenization](preprocessing_and_tokenization), we present our code for preprocessing raw MRI sequences and encoding each volume token via a VQ-VAE. We also provide code for training and evaluating the VQ-VAE model.

(2) Under [Prima training and evaluation](Prima_training_and_evaluation) folder, we present our code for CLIP training of Prima, together with scripts for classification evaluation on prospective test set.

(3) Under [end-to-end inference pipeline](end-to-end_inference_pipeline) folder, we present an end-to-end ready-to-use pipeline for using our model to perform inference on raw, uncurated MRI studies

For detailed instructions, please see the `README.md` file within each folder.

## Data statement

Due to privacy and regulations, we are unable to release raw MRIs from UM-220K. For demonstration purposes, we included scripts for generating fake data (random tensors of the same size as real data) which the model can run on. We use skull-stripped real MRIs on our [Demo Website](https://prima.mlins.org) for protection of patient privacy, although raw data without skull stripping was used for inference and prediction.
