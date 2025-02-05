import torch
from positional_encodings.torch_encodings import PositionalEncoding3D, PositionalEncoding1D
import time
import random


class RachelDatasetPatchifier(torch.nn.Module):
    def __init__(self,in_dim=1024,d=30):
        # in_dim: input dimension
        # d: dimension of 3d positional encoding
        # rel: Private testing, do not use
        super().__init__()
        self.out_dim=in_dim+d
        assert d % 3 == 0
        p_enc = PositionalEncoding3D(d)
        self.p_enc = p_enc(torch.zeros(1,100,100,100,d))[0].view(1000000,d)
        self.p_enc.requires_grad = False
        self.d=d
    def forward(self,xs,coords):
        # xs: the visual token embeddings
        # coords: coordinates of each token
        # This function appends positional encodings to the token
        # engrave: Private testing, do not use



        # this part just deal with legacy code
        if len(self.p_enc) == 100:
            self.p_enc = self.p_enc.view(1000000,self.d)
            self.p_enc.requires_grad=False

        ret = []
        for i,x in enumerate(xs):
            shapes = x.size()
            cdim = torch.zeros(3) # orientation embedding
            if shapes[2] == 2:
                cdim[0] = 1
                div1,div2,div3 = 4,32,32
            elif shapes[3] == 2:
                cdim[1] = 1
                div1,div2,div3 = 32,4,32
                x = x.transpose(2,3)
            else:
                assert shapes[4] == 2
                cdim[2] = 1
                div1,div2,div3 = 32,32,4
                x = x.transpose(2,4)
            divtensor = torch.zeros(3).long()
            divtensor[0] = div1
            divtensor[1] = div2
            divtensor[2] = div3
            divtensor = divtensor.unsqueeze(0)
        
            divcoord = coords[i] // divtensor
            tocatid = divcoord[:,0] * 10000 + divcoord[:,1] * 100 + divcoord[:,2] 
            tocat = self.p_enc[torch.LongTensor(tocatid)]
            cdimcat = cdim.repeat(shapes[0],1)
            ret.append(torch.cat([x.flatten(start_dim=1),tocat.cpu(),cdimcat],dim=1))
        return ret






