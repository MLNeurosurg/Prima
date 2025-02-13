# PRIMA Volumetric Tokenization
![Ext_Data_Fig4_v2 (1)](https://github.com/user-attachments/assets/3488e6db-4bc9-422e-99cb-5027a0e80e90)

**a,** MRI scanners acquire images with specified orientations (e.g. LAS,RAS, etc) and planes (e.g. axial, coronal, sagittal). The MRI tokens will have the same orientation and plane as the sourceMRI sequence after patching. **b,** Examples of VQ-VAE reconstructions in different MRI sequences and orientations. **c,** Because Prima takes as input multiple different orientations and imaging planes, the volume tokenizer should be orientationinvariant, meaning the representation of the same anatomic region should not change if imaged in axial versus coronal plane or LAS versus RAS orientation, for example. We used two strategies: random orientation permutations and 3D-CNN encoders. Our VQ-VAE volume tokenizer is encouraged to encode each volume token equivalently across all orientations under a reconstruction loss. Examples of MRI subvolumes are shown in different orientations after permutation. The latent volume tokens with near-equivalent latent encodings are shown in the center panel. with the reconstructions after the decoder on the right. **d,** Ablation study over the codebook sizes shows the quantization loss validation curves. Larger codebook sizes led to less overfitting and better reconstructions. **e,** Reconstruction losses with and without random orientation permutations.Random permutations regularized the VQ-VAE and resulted in higher-quality reconstructions and lower reconstruction losses. **f,** Examples of reconstructions before and after orientation permutations for different MRI sequences. Reconstructions are perceptually equivalent after forward pass through the VQ-VAE model regardless of orientation or imaging plane. Subtle
reconstruction differences can be seen on difference maps.



# Preprocessing and Tokenization Overview

The initial step in PRIMA's development and implementation consists of MRI preprocessing and tokenization. First, MRI studies composed of various series each (e.g. T1-weighted images, T2-weighted, FLAIR, etc.) are loaded in and converted to LPS orientation. All MRIs are resized to 256 x 256 pixels in the X,Y plane, and slice thickness is converted to 4mm or greater in the Z dimension. A volume tokenization strategy is then used to subdivide each MRI into 32x32x4 (X,Y,Z) patches each of which is compressed 16x in latent space using a variation autoencoder framework with vector-quantization (VQVAE). The VQVAE is made up of 1) a 3D-CNN encoder which encodes each patch down to 8x8x2 volumes with 2 feature dimensions (8x8x2x2), 2) a quantization layer with codebook size of 8192, and 3) a decoder. During training, we additionally implement a random permutation of the image axes to encourage the 3D-CNN to be orientation invariant. Altogether, these approaches allow for a computationally efficient way to train a vision transformer from the latent dimension while preserving high-quality 3D features across image orientations. 


# Preprocessing Steps



# VQVAE Training and Evaluation

Training Details
- The VolumeDataModule in `mrdataset.py` creates batches of image sub-volumes with randomized shapes (e.g. 32x32x4, 4x32x32, 32x4x32)
- Additional random batch permutation is conducted during training in `train.py`

Training Steps
1. Access or update parameters in config file `/configs/vqvae_train_sample_config.yaml`
2. Change directory to train and activate conda environment
3. Run `train.py` to train and validate VQVAE model
   ```
   python train.py -c=/configs/vqvae_train_sample_config.yaml
   ```


