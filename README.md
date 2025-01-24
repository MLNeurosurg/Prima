# Prima: the first general-purpose MRI VLM trained on health system-scale data

This is the official code repository for paper submission "Learning neuroimaging models from health system-scale data", where we introduced Prima, the first general-purpose MRI VLM trained on health system-scale data. This repository includes:

(1) Under `preprocessing and tokenization` folder, we present our code for preprocessing raw MRI sequences and encoding each volume token via a VQ-VAE. We also provide code for training and evaluating the VQ-VAE model.

(2) Under `Prima training and evaluation` folder, we present our code for CLIP training of Prima, together with scripts for classification evaluation on prospective test set.

(3) Under `end-to-end inference pipeline` folder, we present an end-to-end ready-to-use pipeline for using our model to perform inference on raw, uncurated MRI studies

For detailed instructions, please see the `README.md` file within each folder.
