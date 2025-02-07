'''
One script to rule them all! 
This is the main script that will run the end-to-end inference pipeline. This script will:
1. Load the mri study direcotry
2. Minimally preprocess the data
3. Run tokenizer model to get series embeddings
4. Run Prima model to get final predictions
'''

import os
import sys
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import natsort
import nibabel as nib
import pandas as pd
import time
import shutil
import json
import argparse
import logging

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor
from tools.DicomUtils import DicomUtils
from tools.models import ModelLoader


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.study_dir = config['study_dir']
        self.tokenizer_model = config['tokenizer_model']
        self.prima_model = config['prima_model']
        self.output_dir = self.config['output_dir']
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(filename=os.path.join(self.output_dir, 'pipeline.log'), level=logging.INFO)
        logging.info(f'Working on: {self.study_dir}')

    def load_mri_studies(self):
        '''
        Load the MRI study from the study directory
        '''
        logging.info('Loading MRI studies')
        mri_study = DicomUtils(self.study_dir).load_mri_study()
        logging.info('MRI studies loaded')
        return mri_study
    
    def load_tokenizer_model(self, gpu_num=0):
        # load the tokenizer model
        model_config = self.config['model_config']
        tokenizer = ModelLoader(model_config).load_vqvae_model()
        return tokenizer

    def run_tokenizer_model(self, mri_study):
        '''
        Run the tokenizer model to get series embeddings
        '''
        logging.info('Running tokenizer model')
        vqvae = self.load_tokenizer_model().eval().cuda()
        # Run the tokenizer model
        # TODO: create custom tokenizer that vqvae runs on 
        with torch.no_grad():
            series_embeddings = [vqvae(torch.unsqueeze(torch.tensor(series), 0).float().cuda())[0].cpu().numpy() for series in mri_study]

        logging.info('Tokenizer model run')
        return series_embeddings

    def run_prima_model(self, series_embeddings):
        '''
        Run the Prima model to get final predictions
        '''
        logging.info('Running Primamodel')
        # Run the Prima model
        #TODO

        raise NotImplementedError('Working on this')
