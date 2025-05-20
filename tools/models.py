import json
import torch
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from generative.networks.nets import VQVAE
from Prima_training_and_evaluation.model import CLIP


class PrimaModelWHeads(torch.nn.Module):
    """PRIMA model with classification heads for diagnosis, referral, and priority."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PRIMA model with heads.
        
        Args:
            config: Dictionary containing model configuration
        """
        super().__init__()
        # Load CLIP model
        self.clipmodel = torch.load(config['clip_ckpt'],
                                    map_location='cpu').module
        self.clipvisualmodel = self.clipmodel.visual_model
        self.clipvisualmodel.patdis = False

        # Initialize diagnosis and referral heads
        self.diagnosisheads = self._load_heads(config['diagnosis_heads_json'])
        self.referralheads = self._load_heads(config['referral_heads_json'])

        # Create ModuleLists for proper parameter registration
        self.diagnosis_modules = torch.nn.ModuleList(
            [head[0] for head in self.diagnosisheads.values()])
        self.referral_modules = torch.nn.ModuleList(
            [head[0] for head in self.referralheads.values()])

        # Load priority head
        self.priorityhead = torch.load(config['priority_head_ckpt'],
                                       map_location='cpu')

    def _load_heads(self,
                    json_path: str) -> Dict[str, Tuple[torch.nn.Module, int]]:
        """
        Load classification heads from JSON configuration.
        
        Args:
            json_path: Path to the JSON configuration file
            
        Returns:
            Dictionary mapping head names to (model, index) tuples
        """
        heads = {}
        try:
            with open(json_path) as f:
                head_config = json.load(f)

            for name, (_, [(headpath, idx, thresh)]) in head_config.items():
                head = torch.load(headpath, map_location='cpu')
                head.thresh = float(thresh)
                heads[name] = (head, idx)
        except Exception as e:
            raise RuntimeError(f"Failed to load heads from {json_path}: {str(e)}")

        return heads

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing model outputs
        """
        clip_embed = self.clipvisualmodel(x, retpool=True)

        retdict = {
            'diagnosis': {},
            'referral': {},
            'priority': {},
            'clip_emb': clip_embed.detach().cpu() # this can be turned off if not needed
        }

        # Process diagnosis heads
        for name, (head, idx) in self.diagnosisheads.items():
            retdict['diagnosis'][name] = head(clip_embed)[:, idx] - head.thresh

        # Process referral heads
        for name, (head, idx) in self.referralheads.items():
            retdict['referral'][name] = head(clip_embed)[:, idx] - head.thresh

        # Process priority head
        priority_out = self.priorityhead(clip_embed)
        priority_levels = ['none', 'low', 'medium', 'high']
        retdict['priority'] = {
            level: priority_out[:, i]
            for i, level in enumerate(priority_levels)
        }

        return retdict

    @torch.no_grad()
    def forward_one_diag_only(self, x: torch.Tensor,
                              diagname: str) -> torch.Tensor:
        """
        Forward pass for a single diagnosis head.
        
        Args:
            x: Input tensor
            diagname: Name of the diagnosis head to use
            
        Returns:
            Tensor containing the diagnosis output
        """
        if diagname not in self.diagnosisheads:
            raise ValueError(f"Unknown diagnosis head: {diagname}")
            
        clip_embed = self.clipvisualmodel(x, retpool=True)
        head, idx = self.diagnosisheads[diagname]
        return head(clip_embed)[:, idx] - head.thresh

    def make_no_flashattn(self) -> None:
        """Disable flash attention in the visual model."""
        self.clipvisualmodel.make_no_flashattn()


class ModelLoader:
    """Utility class for loading various models."""

    def __init__(self, gpu_num: int = 8, specific_gpu: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            gpu_num: Number of GPUs available
            specific_gpu: Specific GPU to use (if any)
        """
        self.gpu_num = gpu_num
        self.device = torch.device(
            f'cuda:{specific_gpu}' if torch.cuda.is_available() and specific_gpu is not None else 'cpu')
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load_vqvae_model(config: Dict[str, Any]) -> VQVAE:
        """
        Load or initialize a VQVAE model based on configuration.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            VQVAE model instance
            
        Raises:
            FileNotFoundError: If checkpoint file is not found
            ValueError: If configuration is invalid
        """
        try:
            params = config['vqvae_config']
            required_params = [
                "spatial_dims", "in_channels", "out_channels",
                "num_res_layers", "downsample_parameters", "upsample_parameters",
                "num_channels", "num_res_channels", "num_embeddings",
                "embedding_dim"
            ]
            
            # Validate required parameters
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                raise ValueError(f"Missing required parameters: {missing_params}")

            # Initialize the model
            vqvae_model = VQVAE(
                spatial_dims=params["spatial_dims"],
                in_channels=params["in_channels"],
                out_channels=params["out_channels"],
                num_res_layers=params["num_res_layers"],
                downsample_parameters=params["downsample_parameters"],
                upsample_parameters=params["upsample_parameters"],
                num_channels=params["num_channels"],
                num_res_channels=params["num_res_channels"],
                num_embeddings=params["num_embeddings"],
                embedding_dim=params["embedding_dim"],
            )

            # Load pretrained weights if checkpoint path is provided
            if 'ckpt_path' in params and params['ckpt_path']:
                model_path = Path(params['ckpt_path'])
                if not model_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found at {model_path}")

                logging.info(f"Loading pretrained model from {model_path}")
                pl_sd = torch.load(model_path, map_location="cpu")
                vqvae_model.load_state_dict(pl_sd)
            else:
                logging.info("Initializing new VQVAE model with random weights")

            return vqvae_model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load VQVAE model: {str(e)}")

    @staticmethod
    def load_prima_model(config: Dict[str, Any]) -> CLIP:
        """
        Load the PRIMA model.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            CLIP model instance
            
        Raises:
            FileNotFoundError: If checkpoint file is not found
            ValueError: If configuration is invalid
        """
        try:
            model_config = config['prima_config']
            if not model_config:
                raise ValueError("Missing PRIMA model configuration")

            model = CLIP(model_config)

            # Load pretrained weights if specified
            if 'clip_ckpt' in model_config:
                ckpt_path = Path(model_config['clip_ckpt'])
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
                model = torch.load(ckpt_path, map_location='cpu')

            # Move model to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PRIMA model: {str(e)}")

    @staticmethod
    def load_classification_heads(
            config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Union[torch.nn.Module, float]]]:
        """
        Load classification heads from configuration.
        
        Args:
            config: Dictionary containing classification heads configuration
            
        Returns:
            Dictionary mapping condition names to their models and thresholds
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If model files are not found
        """
        try:
            heads_config = config.get('classification_heads', {})
            if not heads_config:
                raise ValueError("No classification heads configuration found")

            heads = {}
            for condition_name, head_info in heads_config.items():
                # Load the classification head model
                head_path = Path(head_info['model_path'])
                if not head_path.exists():
                    raise FileNotFoundError(f"Head model not found at {head_path}")

                head = torch.load(head_path, map_location='cpu')

                # Move to appropriate device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                head = head.to(device)

                # Store the head with its threshold for binary classification
                heads[condition_name] = {
                    'model': head,
                    'threshold': head_info.get('threshold', 0.0)
                }

            return heads
            
        except Exception as e:
            raise RuntimeError(f"Failed to load classification heads: {str(e)}")

    @staticmethod
    def load_full_prima_model(config: Dict[str, Any]) -> PrimaModelWHeads:
        """
        Load the complete PRIMA model including all components.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            Complete PRIMA model instance
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If model files are not found
        """
        try:
            if not config:
                raise ValueError("Empty configuration provided")

            # Create the full model with heads
            full_model = PrimaModelWHeads(config)
            # Move to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            full_model = full_model.to(device)
            
            return full_model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load full PRIMA model: {str(e)}")
