# PRIMA Overview


![1738022426126-795fc098-7897-47f6-941d-8724d4c5638a_1](https://github.com/user-attachments/assets/be841dcb-a446-4e00-b33d-17901c78557f)


Over 220,000 brain MRIs were queried from our health system's picture archiving and communication system (PACS), forming the UM220K dataset. This dataset includes MRI studies from multiple medical centers across the state and the United States. The distribution of MRI counts by county and state is presented. The number of MRIs archived in the PACS system has doubled approximately every six years over the past two decades, highlighting the growing demands on radiology and clinical services. The diagnostic categories reflect the standard operations of a large academic medical center. Prima was trained using a contrastive language-image pre-training (CLIP) framework and a hierarchical vision transformer (ViT) architecture. Full MRI studies were divided into subvolumes, compressed into volume tokens using a tokenizer, and processed by a sequence ViT to extract sequence-level features. Global sequence registers were passed to a study ViT to generate a study-level representation for alignment with radiology reports. Radiology reports were summarized using a large language model (LLM), and a pre-trained neuroradiology language model generated report representations. Finally, the MRI study embeddings and summarized report embeddings were aligned using a CLIP objective.


# Preprocessing and Tokenization Overview

The initial step in PRIMA's development and implementation consists of MRI preprocessing and tokenization. First, MRI studies composed of various series each (e.g. T1-weighted images, T2-weighted, FLAIR, etc.) are loaded in and converted to LPS orientation. All MRI's are resized to 256 x 256 pixels in the X,Y plane, and slice thickness is converted to 4mm or greater in the Z dimension. A volume tokenization strategy is then used to subdivide each MRI into 32x32x4 (X,Y,Z) patches each of which is compressed 16x into latent space using a variation autoencoder framework with vector-quantization (VQVAE). The VQVAE is made up of 1) a 3D-CNN encoder which encodes each patch down to 8x8x2 volumes with 2 feature dimensions (8x8x2x2), 2) a quantization layer with codebook size of 8192, and 3) a decoder. During training, we additionally imnplement a random permutation of the image axes to encourage the 3D-CNN to be orientation invariant. Altogether, these approaches allow for a computationally efficient way to train a vision transformer from the latent dimension while preserving high-quality 3D features across image orientations. 


# Preprocessing Steps



# VQVAE Training and Evaluation

TODO
- [config_parameters]
- [short train/val code lines]
- [include line on permutation]


# VQVAE Inference


