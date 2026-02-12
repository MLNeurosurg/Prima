import json
import pickle
import sys
import torch
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from generative.networks.nets import VQVAE
from tqdm import tqdm
# CLIP imported lazily in load_prima_model() to avoid pulling in transformers until needed

# Repo root (directory containing "tools" and "Prima_training_and_evaluation") for lazy imports
_REPO_ROOT = Path(__file__).resolve().parent.parent


class FullMRIModel(torch.nn.Module):
    """
    Full PRIMA model: CLIP visual encoder + diagnosis, referral, and priority heads.
    Same structure as used for training; can be built from components or loaded from a single checkpoint.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.clipmodel = torch.load(config["clip_ckpt"], map_location="cpu").module
        self.clipvisualmodel = self.clipmodel.visual_model
        self.clipvisualmodel.patdis = False

        self.diagnosisheads = {}
        headjson = json.load(open(config["diagnosis_heads_json"]))
        for name in headjson:
            headpath, idx, thresh = headjson[name][1][0]
            head = torch.load(headpath, map_location="cpu")
            head.thresh = float(thresh)
            self.diagnosisheads[name] = [head, idx]
        self.m1 = torch.nn.ModuleList([self.diagnosisheads[a][0] for a in self.diagnosisheads])

        self.referralheads = {}
        headjson = json.load(open(config["referral_heads_json"]))
        for name in headjson:
            headpath, idx, thresh = headjson[name][1][0]
            head = torch.load(headpath, map_location="cpu")
            head.thresh = float(thresh)
            self.referralheads[name] = [head, idx]
        self.m2 = torch.nn.ModuleList([self.referralheads[a][0] for a in self.referralheads])

        self.priorityhead = torch.load(config["priority_head_ckpt"], map_location="cpu")

    def forward(self, x: Dict[str, Any], inference_only_once = False) -> Dict[str, Any]:
        print("Running CLIP embeddings ...")
        clip_embed = self.clipvisualmodel(x, retpool=True)
        retdict = {
            "diagnosis": {},
            "referral": {},
            "priority": {},
            "clip_emb": clip_embed.detach().cpu(),
        }
        print("Running diagnostic heads ...")
        for name in tqdm(self.diagnosisheads):
            head, idx = self.diagnosisheads[name]
            device_head = head.to(clip_embed.device)
            retdict["diagnosis"][name] = device_head(clip_embed)[:, idx] - head.thresh
            if inference_only_once: # doing this to save GPU memory
                device_head = device_head.cpu()
        print("Running referral heads ...")
        for name in tqdm(self.referralheads):
            head, idx = self.referralheads[name]
            device_head = head.to(clip_embed.device)
            retdict["referral"][name] = device_head(clip_embed)[:, idx] - head.thresh
            if inference_only_once: # doing this to save GPU memory
                device_head = device_head.cpu()
        print("Running priorization heads ...")
        priorityout = self.priorityhead(clip_embed)
        if len(priorityout[0]) == 4:
            retdict["priority"]["none"] = priorityout[:, 0]
            retdict["priority"]["low"] = priorityout[:, 1]
            retdict["priority"]["medium"] = priorityout[:, 2]
            retdict["priority"]["high"] = priorityout[:, 3]
        else:
            retdict["priority"]["none"] = priorityout[:, 0]
            retdict["priority"]["low"] = priorityout[:, 1]
            retdict["priority"]["high"] = priorityout[:, 2]
        return retdict

    def forward_one_diag_only(self, x: Dict[str, Any], diagname: str) -> torch.Tensor:
        clip_embed = self.clipvisualmodel(x, retpool=True)
        head, idx = self.diagnosisheads[diagname]
        return head(clip_embed)[:, idx] - head.thresh

    def make_no_flashattn(self) -> None:
        self.clipvisualmodel.make_no_flashattn()


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
    def load_prima_model(config: Dict[str, Any]):
        """
        Load the PRIMA model (CLIP only).
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            CLIP model instance
        """
        try:
            from Prima_training_and_evaluation.model import CLIP
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
    def load_full_prima_model(config: Dict[str, Any]) -> torch.nn.Module:
        """
        Load the complete PRIMA model (FullMRIModel).
        
        Config may specify:
        - full_model_ckpt: path to a single saved FullMRIModel checkpoint (recommended).
        - Or component keys: clip_ckpt, diagnosis_heads_json, referral_heads_json, priority_head_ckpt.
        
        Returns:
            FullMRIModel instance
        """
        try:
            if not config:
                raise ValueError("Empty configuration provided")

            # Single full-model checkpoint: path is given in config (e.g. from pipeline config file)
            if "full_model_ckpt" in config:
                ckpt_path = Path(config["full_model_ckpt"])
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Full model checkpoint not found at {ckpt_path}")
                logging.info(f"Loading full PRIMA model from {ckpt_path}")
                # Ensure repo root is on path so Prima_training_and_evaluation can be imported
                _repo_root_str = str(_REPO_ROOT)
                if _repo_root_str not in sys.path:
                    sys.path.insert(0, _repo_root_str)
                # Allow checkpoints saved under different module paths to deserialize
                this_module = sys.modules["tools.models"]
                sys.modules["complete_visual_model"] = this_module
                _main = sys.modules.get("__main__")
                _saved_main_fullmri = getattr(_main, "FullMRIModel", None) if _main is not None else None
                if _main is not None:
                    setattr(_main, "FullMRIModel", FullMRIModel)
                _saved_model = sys.modules.get("model")
                _saved_model_parts = sys.modules.get("model_parts")
                _saved_patchify = sys.modules.get("patchify")
                # Custom unpickler: resolve 'model' and 'model_parts' to Prima_training_and_evaluation submodules on demand
                def _prima_find_class(mod_name, name):
                    if mod_name == "model":
                        import Prima_training_and_evaluation.model as _m
                        sys.modules["model"] = _m
                        return getattr(_m, name)
                    if mod_name == "model_parts":
                        import Prima_training_and_evaluation.model_parts as _mp
                        sys.modules["model_parts"] = _mp
                        return getattr(_mp, name)
                    if mod_name == "patchify":
                        import Prima_training_and_evaluation.patchify as _pf
                        sys.modules["patchify"] = _pf
                        return getattr(_pf, name)
                    return None

                class _PrimaUnpickler(pickle.Unpickler):
                    def find_class(self, mod_name, name):
                        res = _prima_find_class(mod_name, name)
                        if res is not None:
                            return res
                        return super().find_class(mod_name, name)

                _prima_pickle = type(sys)("pickle")
                _prima_pickle.Unpickler = _PrimaUnpickler
                for _attr in ("loads", "load", "dumps", "dump", "Pickler", "HIGHEST_PROTOCOL", "DEFAULT_PROTOCOL"):
                    if hasattr(pickle, _attr):
                        setattr(_prima_pickle, _attr, getattr(pickle, _attr))
                try:
                    full_model = torch.load(
                        str(ckpt_path), map_location="cpu", weights_only=False, pickle_module=_prima_pickle
                    )
                finally:
                    if sys.modules.get("complete_visual_model") is this_module:
                        del sys.modules["complete_visual_model"]
                    if _saved_model is not None:
                        sys.modules["model"] = _saved_model
                    elif "model" in sys.modules and sys.modules["model"].__name__ == "Prima_training_and_evaluation.model":
                        del sys.modules["model"]
                    if _saved_model_parts is not None:
                        sys.modules["model_parts"] = _saved_model_parts
                    elif "model_parts" in sys.modules and sys.modules["model_parts"].__name__ == "Prima_training_and_evaluation.model_parts":
                        del sys.modules["model_parts"]
                    if _saved_patchify is not None:
                        sys.modules["patchify"] = _saved_patchify
                    elif "patchify" in sys.modules and sys.modules["patchify"].__name__ == "Prima_training_and_evaluation.patchify":
                        del sys.modules["patchify"]
                    if _main is not None:
                        if _saved_main_fullmri is not None:
                            setattr(_main, "FullMRIModel", _saved_main_fullmri)
                        elif hasattr(_main, "FullMRIModel") and getattr(_main, "FullMRIModel") is FullMRIModel:
                            delattr(_main, "FullMRIModel")
                if hasattr(full_model, "module"):
                    full_model = full_model.module
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                full_model = full_model.to(device)
                return full_model

            # Build from components
            full_model = FullMRIModel(config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            full_model = full_model.to(device)
            return full_model

        except Exception as e:
            msg = str(e)
            hint = ""
            if "No module named " in msg:
                import re
                m = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
                if m:
                    hint = f" Try: pip install {m.group(1)}"
            raise RuntimeError(f"Failed to load full PRIMA model: {msg}{hint}")
