import json
import torch
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from generative.networks.nets import VQVAE
from Prima_training_and_evaluation.model import CLIP

class PrimaModelWHeads(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        # Load CLIP model
        self.clipmodel = torch.load(config['clip_ckpt'], map_location='cpu').module
        self.clipvisualmodel = self.clipmodel.visual_model
        self.clipvisualmodel.patdis = False

        # Initialize diagnosis and referral heads
        self.diagnosisheads = self._load_heads(config['diagnosis_heads_json'])
        self.referralheads = self._load_heads(config['referral_heads_json'])
        
        # Create ModuleLists for proper parameter registration
        self.diagnosis_modules = torch.nn.ModuleList([head[0] for head in self.diagnosisheads.values()])
        self.referral_modules = torch.nn.ModuleList([head[0] for head in self.referralheads.values()])
        
        # Load priority head
        self.priorityhead = torch.load(config['priority_head_ckpt'], map_location='cpu')

    def _load_heads(self, json_path: str) -> Dict[str, Tuple[torch.nn.Module, int]]:
        """Load classification heads from JSON configuration."""
        heads = {}
        with open(json_path) as f:
            head_config = json.load(f)
            
        for name, (_, [(headpath, idx, thresh)]) in head_config.items():
            head = torch.load(headpath, map_location='cpu')
            head.thresh = float(thresh)
            heads[name] = (head, idx)
            
        return heads

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass through the model."""
        clip_embed = self.clipvisualmodel(x, retpool=True)
        
        retdict = {
            'diagnosis': {},
            'referral': {},
            'priority': {},
            'clip_emb': clip_embed.detach().cpu()
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
        retdict['priority'] = {level: priority_out[:, i] for i, level in enumerate(priority_levels)}

        return retdict

    @torch.no_grad()
    def forward_one_diag_only(self, x: torch.Tensor, diagname: str) -> torch.Tensor:
        """Forward pass for a single diagnosis head."""
        clip_embed = self.clipvisualmodel(x, retpool=True)
        head, idx = self.diagnosisheads[diagname]
        return head(clip_embed)[:, idx] - head.thresh

    def make_no_flashattn(self) -> None:
        """Disable flash attention in the visual model."""
        self.clipvisualmodel.make_no_flashattn()



class ModelLoader:
    def __init__(self, gpu_num: int = 8):
        self.gpu_num = gpu_num
        self.device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def load_vqvae_model(self,config: Dict) -> VQVAE:
        """Load or initialize a VQVAE model based on configuration.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            VQVAE model instance
        """
        params = config['vqvae_config']
        
        # Initialize the model with configuration parameters
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
                
            print(f"Loading pretrained model from {model_path}")
            pl_sd = torch.load(model_path, map_location="cpu")
            vqvae_model.load_state_dict(pl_sd)
        else:
            print("Initializing new VQVAE model with random weights")

        # Move model to appropriate device
        vqvae_model.to(self.device)
        return vqvae_model
    
    @staticmethod
    def load_prima_model(config: Dict) -> CLIP:

        model_config = config['prima_config']

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
    
    @staticmethod
    def load_classification_heads(config: Dict) -> Dict[str, Dict[str, Union[torch.nn.Module, float]]]:
        """Load classification heads from configuration.
        
        Args:
            config: Dictionary containing classification heads configuration
            
        Returns:
            Dictionary mapping condition names to their models and thresholds
        """
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
    
    @staticmethod
    def load_full_prima_model(config: Dict) -> torch.nn.Module:
        """Load the complete PRIMA model including all components.
        
        Args:
            config: Dictionary containing full model configuration with the following keys:
                - clip_ckpt: Path to CLIP model checkpoint
                - diagnosis_heads_json: Path to diagnosis heads configuration
                - referral_heads_json: Path to referral heads configuration
                - priority_head_ckpt: Path to priority head checkpoint
                
        Returns:
            FullMRIModel instance with all components loaded
        """        
        # Validate required configuration keys
        required_keys = ['clip_ckpt', 'diagnosis_heads_json', 'referral_heads_json', 'priority_head_ckpt']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
            
        # Validate file paths
        for key in required_keys:
            path = Path(config[key])
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Initialize the full model
        model = PrimaModelWHeads(config)
        
        # Move model to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        return model
    
    
    