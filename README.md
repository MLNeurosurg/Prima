# Prima: general-purpose MRI VLM trained on health system-scale data



This is the official code repository for paper submission "Learning neuroimaging models from health system-scale data", where we introduced Prima, the first general-purpose MRI VLM trained on health system-scale data. [Demo Website](https://prima.mlins.org)

```
Preprint coming soon!
```

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
using a VQ-VAE encoder. We provide code for preprocessing and VQ-VAE tokenization under [preprocessing and tokenization](preprocessing%20and%20tokenization).

The latent VQ-VAE tokens are then passed forward to the sequence ViT
with the concatenated positional encodings. **d,** The hierarchical ViT is trained using a CLIP objective
on frozen volume token features. The sequence ViT is a multimodal transformer that takes as input
both the volume tokens and the embedded free-text sequence description. The series registers are
passed forward to the study ViT that outputs a single representation for the full MRI study. The paired
reports are summarized and passed through a pre-trained neuroradiology model to align the MRI
study and the paired report. **e,** A transfer learning strategy is used such that the volume tokenizer,
sequence and study transformers are frozen, and an MLP is trained on the learned study features for
radiologic and clinical task prediction. We provide code for training the hierarchical vision transformer using CLIP objective, as well as code for training the task-specific MLPs and evaluating on prospective test set, under [Prima training and evaluation](Prima%20training%20and%20evaluation).

We also provide an **end-to-end ready-to-use** inference pipeline for applying Prima directly on raw MRI scans at [end-to-end inference pipeline](end-to-end%20inference%20pipeline).

# Repository Structure

This repository provides code and instructions for 3 parts of our project

(1) Under [preprocessing and tokenization](preprocessing%20and%20tokenization), we present our code for preprocessing raw MRI sequences and encoding each volume token via a VQ-VAE. We also provide code for training and evaluating the VQ-VAE model.

(2) Under [Prima training and evaluation](Prima%20training%20and%20evaluation) folder, we present our code for CLIP training of Prima, together with scripts for classification evaluation on prospective test set.

(3) Under [end-to-end inference pipeline](end-to-end%20inference%20pipeline) folder, we present an end-to-end ready-to-use pipeline for using our model to perform inference on raw, uncurated MRI studies

For detailed instructions, please see the `README.md` file within each folder.

## Data statement

Due to privacy and regulations, we are unable to release raw MRIs from UM-220K. For demonstration purposes, we included scripts for generating fake data (random tensors of the same size as real data) which the model can run on. We use skull-stripped real MRIs on our [Demo Website](https://prima.mlins.org) for protection of patient privacy, although raw data without skull stripping was used for inference and prediction.
