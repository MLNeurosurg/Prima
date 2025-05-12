import json
import torch
import sys,os
sys.path.insert(0,os.getcwd())
from dataset import RachelDataset, collatevisualhash
import copy
from tqdm import tqdm

from lime import lime_base
from scipy import spatial
from sklearn.utils.validation import check_random_state
import numpy as np



# Tools for figuring out the series and head models
def getseries(h):
    data = datadict[h]
    return [r[0] for r in data[1]]
    
def getheadinfo(configjson,taskname):
    infos = json.load(open(configjson))
    return infos[taskname]

class Lime_Explainer:
    def __init__(self, kernelfn=None, feature_selection="none", verbose=False):
        if kernelfn is None:

            def kernelfn(d):
                return np.sqrt(np.exp(-(d**2) / 0.25**2))

        self.base = lime_base.LimeBase(kernelfn, verbose)
        self.fs = feature_selection

    def explain_instance(
        self, inp, serie_of_interest, classfn, num_samples, seed=0, fracs=1
    ):
        samples = num_samples
        randomstate = check_random_state(seed)
        series_ord = inp['serienamestr'].index(serie_of_interest)
        lentokens = len(inp['visual'][series_ord])
        
        masks = (
            randomstate.randint(0, fracs + 1, lentokens*samples)
            .reshape(samples, lentokens)
            .astype(np.float64)
        )
        masks /= float(fracs)
        masks[0] = 1
        for i,mask in enumerate(masks):
            if np.sum(mask) == 0.0:
                masks[i] = 1
        # print(samples)
        distances = np.zeros(samples)
        llabels = np.zeros((samples, 1))
        for i in tqdm(range(samples)):
            newdata = copy.deepcopy(inp)
            tensormask = torch.LongTensor(masks[i])
            indices = torch.nonzero(tensormask)[:,0]
            newdata['visual'][series_ord] = newdata['visual'][series_ord][indices]
            newdata['coordinates'][series_ord] = newdata['coordinates'][series_ord][indices]
            llabels[i,0] = classfn([newdata])

        ret = self.base.explain_instance_with_data(
            masks, llabels, distances, 0, lentokens, feature_selection=self.fs
        )
        return ret

def getlogits(datas):
    collated = collate(datas)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
            clipout = visualclip(collated,retpool=True)
    finalout = head(clipout)
    outval = finalout[:,classid]
    return outval