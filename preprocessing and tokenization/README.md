# Preprocessing and Tokenization Overview

The initial step in PRIMA's development and implementation consists of MRI preprocessing and tokenization. First, MRI studies composed of various series each (e.g. T1-weighted images, T2-weighted, FLAIR, etc.) are loaded in and converted to LPS orientation. All MRI's are resized to 256 x 256 pixels in the X,Y plane, and slice thickness is converted to 4mm or greater in the Z dimension. A volume tokenization strategy is then used to subdivide each MRI into 32x32x4 (X,Y,Z) patches each of which is compressed 16x into latent space of size 8x8x2x2 using a variation autoencoder framework with vector-quantization (VQVAE). This approaches allows for a computationally efficient way to train a vision transformer from the latent dimension while preserving high-quality 3D features from each image. 


# Preprocessing Steps



# VQVAE Training and Evaluation
