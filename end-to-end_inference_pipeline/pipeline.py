'''
One script to rule them all! 
This is the main script that will run the end-to-end inference pipeline. 
This script will:
1. Load the mri study direcotry
2. Minimally preprocess the data
3. Run tokenizer model to get series embeddings
4. Run Prima model to provide final classification, acuity and referral recommendations

Assumptions:
1. this pipeline is run on a single study at a time. *can be extended to multiple studies with some modifications*
2. the vqvae, prima and classification heads are already trained
2a. Prima model and all the heads are combined into a single model
3. the config file needs to be passed in as an argument
3a. the config file should have the following keys:
    - study_dir: path to the study directory
    - output_dir: path to the output directory
    - prima_model_config: path to the prima model config file
    - vqvae_config: path to the vqvae config file
'''

import os
import SimpleITK as sitk
import torch 
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import gc
import psutil
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
from Prima_training_and_evaluation.dataset import MrDataset
from tools.DicomUtils import DicomUtils
from tools.models import ModelLoader
from tools.mrcommondataset import MrVoxelDataset
from tools.utilities import chartovec


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    study_dir: str
    output_dir: str
    tokenizer_model_config: str
    prima_model_config: str
    batch_size: int = 1
    num_workers: int = 2
    max_tokens_per_chunk: int = 400
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create a PipelineConfig from a dictionary."""
        required_keys = ['study_dir', 'output_dir', 'tokenizer_model_config', 'prima_model_config']
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        return cls(**config_dict)


class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = PipelineConfig.from_dict(config)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f'Initializing pipeline with config: {self.config}')
        
        # Initialize models as None
        self.tokenizer_model = None
        self.prima_model = None

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.output_dir / 'pipeline.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        if self.tokenizer_model is not None:
            del self.tokenizer_model
        if self.prima_model is not None:
            del self.prima_model
        torch.cuda.empty_cache()
        gc.collect()

    def load_mri_study(self) -> Tuple[List[sitk.Image], List[str]]:
        """
        Load the MRI study from the study directory.
        
        Returns:
            Tuple containing list of MRI images and series names
        """
        self.logger.info('Loading MRI study')
        try:
            self.mri_study, self.series_list = DicomUtils.load_mri_study(self.config.study_dir)
            self.logger.info(f'Successfully loaded {len(self.mri_study)} series')
            return self.mri_study, self.series_list
        except Exception as e:
            self.logger.error(f'Failed to load MRI study: {str(e)}')
            raise
    
    def load_tokenizer_model(self) -> torch.nn.Module:
        """Load the tokenizer model."""
        if self.tokenizer_model is None:
            self.logger.info('Loading tokenizer model')
            try:
                self.tokenizer_model = ModelLoader.load_vqvae_model(self.config.tokenizer_model_config)
                self.tokenizer_model = self.tokenizer_model.to(self.config.device)
                self.tokenizer_model.eval()
            except Exception as e:
                self.logger.error(f'Failed to load tokenizer model: {str(e)}')
                raise
        return self.tokenizer_model
    
    def load_full_prima_model(self) -> torch.nn.Module:
        """Load the full Prima model."""
        if self.prima_model is None:
            self.logger.info('Loading Prima model')
            try:
                self.prima_model = ModelLoader.load_full_prima_model(self.config.prima_model_config)
                self.prima_model = self.prima_model.to(self.config.device)
                self.prima_model.eval()
            except Exception as e:
                self.logger.error(f'Failed to load Prima model: {str(e)}')
                raise
        return self.prima_model
    
    def create_dataset(self, mri_study: List[sitk.Image]) -> DataLoader:
        """
        Create a dataset from the MRI study.
        
        Args:
            mri_study: List of MRI images
            
        Returns:
            DataLoader for the dataset
        """
        try:
            dataset = MrVoxelDataset(mri_study)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
            return dataloader
        except Exception as e:
            self.logger.error(f'Failed to create dataset: {str(e)}')
            raise

    def prepare_prima_input(self) -> Dict[str, Any]:
        """
        Prepare the input for the Prima model.
        
        Returns:
            Dictionary containing model inputs
        """
        try:
            # load the study
            mri_study, series_names = self.load_mri_study()

            # run the tokenizer model
            series_embeddings = self.run_tokenizer_model(mri_study)
            
            # prepare the input for the prima model
            visuals = [torch.stack(emb) for emb in series_embeddings]
            
            # Create series name tensors
            serienames = torch.tensor([chartovec(name) for name in series_names], dtype=torch.long)
            
            # Create study description tensor
            study_desc = torch.tensor([ord(c) for c in "MRI BRAIN"], dtype=torch.long)
            
            # Create lengths tensors
            study_lens = torch.tensor([len(series_embeddings)], dtype=torch.long)
            serie_lenss = torch.tensor([[len(v) for v in visuals]], dtype=torch.long)
            
            return {
                'visual': visuals,
                'lens': study_lens,
                'lenss': serie_lenss,
                'hash': ["study_0"],
                'serienames': serienames,
                'studydescription': study_desc.unsqueeze(0)
            }
        except Exception as e:
            self.logger.error(f'Failed to prepare Prima input: {str(e)}')
            raise

    def run_tokenizer_model(self, mri_study: List[sitk.Image]) -> List[torch.Tensor]:
        """
        Run the tokenizer model to get series embeddings.
        
        Args:
            mri_study: List of MRI images
            
        Returns:
            List of series embeddings
        """
        self.logger.info('Running tokenizer model')
        vqvae = self.load_tokenizer_model()
        dataloader = self.create_dataset(mri_study)
        series_embeddings = []
        
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Processing series"):
                    token_list = []
                    tokens = batch[0]
                    num_tokens = tokens.shape[0]
                    
                    # Split into chunks
                    num_chunks = (num_tokens + self.config.max_tokens_per_chunk - 1) // self.config.max_tokens_per_chunk
                    
                    for j in range(num_chunks):
                        start_idx = j * self.config.max_tokens_per_chunk
                        end_idx = min((j + 1) * self.config.max_tokens_per_chunk, num_tokens)
                        chunk = tokens[start_idx:end_idx].unsqueeze(0)
                        token_list.append(chunk)
                    
                    # Get embeddings
                    embeddings = [
                        vqvae.encode(chunk.to(self.config.device)).detach().cpu() 
                        for chunk in token_list
                    ]
                    series_embedding = torch.cat(embeddings, dim=1)
                    series_embeddings.append(series_embedding)
                    
            self.logger.info('Tokenizer model run complete')
            return series_embeddings
        except Exception as e:
            self.logger.error(f'Failed to run tokenizer model: {str(e)}')
            raise
        finally:
            self.tokenizer_model = None
            torch.cuda.empty_cache()
            gc.collect()

    def run_prima_model(self, series_embeddings: List[torch.Tensor], series_names: List[str]) -> Dict[str, Any]:
        """
        Run the Prima model to get final predictions.
        
        Args:
            series_embeddings: List of series embeddings
            series_names: List of series names
            
        Returns:
            Dictionary containing model predictions
        """
        self.logger.info('Running Prima model')
        
        try:
            # Prepare input for Prima model
            prima_input = self.prepare_prima_input()

            # Move input to correct device
            prima_input = {
                k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                for k, v in prima_input.items()
            }

            # Load model if not already loaded
            if self.prima_model is None:
                self.prima_model = self.load_full_prima_model()            
            
            # Run inference
            with torch.no_grad(), torch.amp.autocast(device_type=self.config.device, dtype=torch.float16):
                predictions = self.prima_model(prima_input)
                    
            # Save predictions
            output_path = self.output_dir / 'predictions.json'
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
                
            self.logger.info(f'Predictions saved to {output_path}')
            return predictions
        except Exception as e:
            self.logger.error(f'Failed to run Prima model: {str(e)}')
            raise
        finally:
            self._cleanup()


if __name__=="__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='End-to-end inference pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    args = parser.parse_args()

    try:
        # Load config
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Initialize pipeline
        pipeline = Pipeline(config)
        
        # Run pipeline steps
        pipeline.logger.info("Starting pipeline execution")
        
        # Step 1: Load MRI study
        mri_study, series_names = pipeline.load_mri_study()
        pipeline.logger.info(f"Loaded {len(mri_study)} series from study")
        
        # Step 2: Run tokenizer model
        series_embeddings = pipeline.run_tokenizer_model(mri_study)
        pipeline.logger.info(f"Generated embeddings for {len(series_embeddings)} series")
        
        # Step 3: Run Prima model
        predictions = pipeline.run_prima_model(series_embeddings, series_names)
        pipeline.logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise
