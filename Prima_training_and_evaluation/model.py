import torch
from typing import Dict, Tuple, Union, Optional
from transformers import GPT2Model
from .model_parts import ViT, GPTWrapper, HierViT, SerieTransformerEncoder
from .patchify import MedicalImagePatchifier


class CLIP(torch.nn.Module):
    """Contrastive Language-Image Pre-training (CLIP) model for medical imaging.
    
    This model combines a visual encoder (HierViT) with a text encoder (GPT2) to learn
    joint representations of medical images and their corresponding text descriptions.
    """

    def __init__(self, config: Dict):
        """Initialize the CLIP model.
        
        Args:
            config: Configuration dictionary containing:
                - data: Data configuration (in_dim, d)
                - model: Model configuration for text and visual components
                - train: Training configuration (optional temperature parameters)
        """
        super(CLIP, self).__init__()
        self.config = config
        dataconfig = config['data']
        modelconfig = config['model']

        # Initialize temperature parameters
        self.temperature = self._init_temperature(config, 'init_temperature')
        self.patdistemperature = self._init_temperature(
            config, 'patdis_init_temperature')

        # Initialize patchifier
        self.patchifier = MedicalImagePatchifier(dataconfig['in_dim'],
                                                 dataconfig['d'])

        # Initialize text model
        self.text_model = self._init_text_model(modelconfig)

        # Initialize visual model
        self.visual_model = self._init_visual_model(modelconfig, config)

        # Initialize loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def _init_temperature(self, config: Dict,
                          param_name: str) -> torch.nn.Parameter:
        """Initialize temperature parameter if specified in config."""
        temp = torch.zeros(1)
        if param_name in config['train']:
            temp[0] = config['train'][param_name]
            return torch.nn.Parameter(temp)
        return temp

    def _init_text_model(self, modelconfig: Dict) -> GPTWrapper:
        """Initialize the text model component."""
        if modelconfig['text']['type'] != 'gpt2':
            raise NotImplementedError("Only GPT2 text model is supported")

        if 'ckpt_path' in modelconfig['text']:
            ckpt_path = modelconfig['text']['ckpt_path']
            if ckpt_path.endswith('.pt'):
                text_model = torch.load(ckpt_path)
            else:
                text_model = GPT2Model.from_pretrained(ckpt_path)
        else:
            text_model = GPT2Model.from_pretrained('gpt2')

        return GPTWrapper(text_model, modelconfig['feature_dim'], 768)

    def _init_visual_model(self, modelconfig: Dict, config: Dict) -> HierViT:
        """Initialize the visual model component."""
        if modelconfig['visual']['type'] != 'hiervit':
            raise NotImplementedError("Only HierViT visual model is supported")

        if 'ckpt_path' in modelconfig['visual']:
            return torch.load(modelconfig['visual']['ckpt_path'])

        innerconfig = modelconfig['visual']['inner']
        if not innerconfig.get('dim'):
            innerconfig['dim'] = self.patchifier.out_dim

        outerconfig = modelconfig['visual']['outer']
        visual_config = {
            'useseriename':
            'useseriename' in modelconfig['visual'],
            'usestudydescription':
            'usestudydescription' in modelconfig['visual'],
            'patdis':
            'patient_series_discrimination' in config['train'],
            'pretrainedserieencoder':
            modelconfig['visual'].get('serie_encoder_ckpt')
        }

        return HierViT(innerconfig, outerconfig, **visual_config)

    def forward(
        self,
        batch: Dict,
        visualonly: bool = False,
        textonly: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            batch: Input batch containing text and/or image data
            visualonly: If True, only process visual input
            textonly: If True, only process text input
            
        Returns:
            Either text embeddings, image embeddings, or both depending on mode
        """
        if textonly or (hasattr(self, 'textonly') and self.textonly):
            text_encoded = self.text_model(batch['text'], batch['textlen'])
            return self.unitize(text_encoded)

        image_encoded = self.visual_model(batch)
        if visualonly or (hasattr(self, 'visualonly') and self.visualonly):
            return self.unitize(image_encoded)

        text_encoded = self.text_model(batch['text'], batch['textlen'])
        return self.unitize(text_encoded), self.unitize(image_encoded)

    def unitize(
            self, vecs: Union[torch.Tensor,
                              Tuple]) -> Union[torch.Tensor, Tuple]:
        """Normalize vectors to unit length.
        
        Args:
            vecs: Input vectors or tuple of vectors to normalize
            
        Returns:
            Normalized vectors maintaining the same structure as input
        """
        if isinstance(vecs, tuple):
            if len(vecs) == 3:
                a, b, c = vecs
                return self.unitize(a), self.unitize(b), c
            elif len(vecs) == 2:
                a, b = vecs
                return a, self.unitize(b)

        norms = torch.norm(vecs, dim=1, keepdim=True)
        return vecs / norms


class SerieCLIP(torch.nn.Module):
    """Series-based CLIP model for medical imaging.
    
    This model processes medical image series with their corresponding series names,
    using a ViT for visual encoding and a transformer for text encoding.
    """

    def __init__(self, config: Dict):
        """Initialize the SerieCLIP model.
        
        Args:
            config: Configuration dictionary containing:
                - data: Data configuration (in_dim, d)
                - model: Model configuration for text and visual components
                - train: Training configuration (optional temperature parameters)
        """
        super(SerieCLIP, self).__init__()
        self.config = config
        dataconfig = config['data']
        modelconfig = config['model']

        # Initialize temperature parameter
        self.temperature = self._init_temperature(config)

        # Initialize patchifier
        self.patchifier = MedicalImagePatchifier(dataconfig['in_dim'],
                                                 dataconfig['d'])

        # Initialize visual model
        self.visual_model = self._init_visual_model(modelconfig)

        # Initialize text model
        self.text_model = SerieTransformerEncoder(modelconfig['feature_dim'])

    def _init_temperature(self, config: Dict) -> torch.nn.Parameter:
        """Initialize temperature parameter if specified in config."""
        temp = torch.zeros(1)
        if 'init_temperature' in config['train']:
            temp[0] = config['train']['init_temperature']
            return torch.nn.Parameter(temp)
        return temp

    def _init_visual_model(self, modelconfig: Dict) -> ViT:
        """Initialize the visual model component."""
        if 'ckpt_path' in modelconfig['visual']:
            return torch.load(modelconfig['visual']['ckpt_path'])

        return ViT(dim=self.patchifier.out_dim,
                   num_classes=modelconfig['feature_dim'],
                   depth=modelconfig['visual']['depth'],
                   heads=modelconfig['visual']['heads'],
                   mlp_dim=modelconfig['visual']['mlp_dim'],
                   dim_head=modelconfig['visual']['dim_head'],
                   clsnum=modelconfig['visual']['clsnum'])

    def forward(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input dictionary containing image data and series names
            
        Returns:
            Tuple of (text_embeddings, image_embeddings)
        """
        image_encoded = self.visual_model(x)
        text_encoded = self.text_model(x['serienames'])
        return self.unitize(text_encoded), self.unitize(image_encoded)

    def unitize(
            self, vecs: Union[torch.Tensor,
                              Tuple]) -> Union[torch.Tensor, Tuple]:
        """Normalize vectors to unit length.
        
        Args:
            vecs: Input vectors or tuple of vectors to normalize
            
        Returns:
            Normalized vectors maintaining the same structure as input
        """
        if isinstance(vecs, tuple):
            a, b, c = vecs
            return self.unitize(a), self.unitize(b), c

        norms = torch.norm(vecs, dim=1, keepdim=True)
        return vecs / norms
