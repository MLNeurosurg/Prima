import torch
from transformers import GPT2LMHeadModel,GPT2Model
from model_parts import ViT,GPTWrapper,HierViT,SerieTransformerEncoder
from patchify import RachelDatasetPatchifier


# The entire CLIP model
class CLIP(torch.nn.Module):
    def __init__(self, config):
        super(CLIP, self).__init__()
        self.config = config
        dataconfig = config['data']
        modelconfig = config['model']
        a = torch.zeros(1)
        if 'init_temperature' in config['train']:
            a[0] = config['train']['init_temperature']
            self.temperature = torch.nn.Parameter(a)
        else:
            self.temperature = a

        a = torch.zeros(1)
        if 'patdis_init_temperature' in config['train']:
            a[0] = config['train']['patdis_init_temperature']
            self.patdistemperature = torch.nn.Parameter(a)
        else:
            self.patdistemperature = a
        


        # decide which patchifier to use
        self.patchifier = RachelDatasetPatchifier(dataconfig['in_dim'],dataconfig['d'])
        
        # initialize language model
        if modelconfig['text']['type'] == 'gpt2':
            if 'ckpt_path' in modelconfig['text']:
                if modelconfig['text']['ckpt_path'][:-3] == '.pt':
                    self.text_model = torch.load(modelconfig['text']['ckpt_path'])
                else:
                    self.text_model = GPT2Model.from_pretrained(modelconfig['text']['ckpt_path'])
            else:
                self.text_model = GPT2Model.from_pretrained('gpt2')
            self.text_model = GPTWrapper(self.text_model,modelconfig['feature_dim'],768)
        else:
            raise NotImplementedError
        
        # initialize visual model
        if modelconfig['visual']['type'] == 'hiervit':
            if 'ckpt_path' in modelconfig['visual']:
                self.visual_model = torch.load(modelconfig['visual']['ckpt_path'])
            else:
                innerconfig = modelconfig['visual']['inner']
                if not innerconfig.get('dim'):
                    innerconfig['dim'] = self.patchifier.out_dim
                outerconfig = modelconfig['visual']['outer']
                useseriename = 'useseriename' in modelconfig['visual']
                usestudydescription = 'usestudydescription' in modelconfig['visual']
                pretrainedserieencoder = None
                if 'serie_encoder_ckpt'  in modelconfig['visual']:
                    pretrainedserieencoder = modelconfig['visual']['serie_encoder_ckpt']
                self.visual_model = HierViT(innerconfig,outerconfig,useseriename=useseriename,usestudydescription=usestudydescription,patdis='patient_series_discrimination' in config['train'],pretrainedserieencoder = pretrainedserieencoder)
        else:
            raise NotImplementedError
        

        self.criterion = torch.nn.CrossEntropyLoss()
        # self.visual_model.make_no_flashattn()  # uncomment this if your GPU does not support flash attention or if you do not wish to use flash attention

    def forward(self, batch, visualonly = False, textonly = False):
        if textonly or (hasattr(self,'textonly') and self.textonly):
            text_encoded = self.text_model(batch['text'],batch['textlen'])
            return self.unitize(text_encoded)
        image_encoded = self.visual_model(batch)
        if visualonly or (hasattr(self,'visualonly') and self.visualonly):
            return self.unitize(image_encoded)
        text_encoded = self.text_model(batch['text'],batch['textlen'])
        return self.unitize(text_encoded),self.unitize(image_encoded)
    def unitize(self,vecs): # unitize each vector
        if isinstance(vecs,tuple):
            if len(vecs) == 3:
                a,b,c = vecs
                return self.unitize(a),self.unitize(b),c
            elif len(vecs) == 2:
                a,b = vecs
                return a,self.unitize(b)
        norms = torch.norm(vecs,dim=1, keepdim=True)
        return vecs/norms


# Serie CLIP model
class SerieCLIP(torch.nn.Module):
    def __init__(self, config):
        super(SerieCLIP, self).__init__()
        self.config = config
        dataconfig = config['data']
        modelconfig = config['model']
        a = torch.zeros(1)
        if 'init_temperature' in config['train']:
            a[0] = config['train']['init_temperature']
            self.temperature = torch.nn.Parameter(a)
        else:
            self.temperature = a

        # decide which patchifier to use
        self.patchifier = RachelDatasetPatchifier(dataconfig['in_dim'],dataconfig['d'])

        if 'ckpt_path' in modelconfig['visual']:
            self.visual_model = torch.load(modelconfig['visual']['ckpt_path'])
        else:
            self.visual_model = ViT(
                    dim = self.patchifier.out_dim,
                    num_classes = modelconfig['feature_dim'],
                    depth = modelconfig['visual']['depth'],
                    heads = modelconfig['visual']['heads'],
                    mlp_dim = modelconfig['visual']['mlp_dim'],
                    dim_head = modelconfig['visual']['dim_head'],
                    clsnum = modelconfig['visual']['clsnum'])
        
        self.text_model = SerieTransformerEncoder(modelconfig['feature_dim'])

    def forward(self,x):
        image_encoded = self.visual_model(x)
        text_encoded = self.text_model(x['serienames'])
        return self.unitize(text_encoded),self.unitize(image_encoded)
    def unitize(self,vecs): # unitize each vector
        if isinstance(vecs,tuple):
            a,b,c = vecs
            return self.unitize(a),self.unitize(b),c
        norms = torch.norm(vecs,dim=1, keepdim=True)
        return vecs/norms

        







