import torch
from generative.networks.nets import VQVAE
import json



class ModelLoader:
    def __init__(self, model_config, gpu_num):
        self.model_config = model_config
        self.gpu_num = gpu_num

    def load_vqvae_model(self):
        params = self.model_config['vqvae']
        model_path = params['ckpt_path']

        vqvae_model = VQVAE(
            spatial_dims=params["spatial_dims"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            num_res_layers=params["num_res_layers"],
            downsample_parameters=params[
                "downsample_parameters"],  # stride ks dilation padding
            upsample_parameters=params["upsample_parameters"],  # output padding
            num_channels=params["num_channels"],
            num_res_channels=params["num_res_channels"],
            num_embeddings=params["num_embeddings"],  # 8192
            embedding_dim=params["embedding_dim"],
        )
        print(f"Loading model from {model_path}")
        pl_sd = torch.load(model_path, map_location="cpu")
        vqvae_model.load_state_dict(pl_sd)  #, strict=False
        vqvae_model.to(torch.device(f'cuda:{self.gpu_num}'))
        return vqvae_model
    
    def load_prima_model(self):
        raise NotImplementedError('Working on this')
    
    def load_classfication_heads(self):
        raise NotImplementedError('Working on this')
    
    