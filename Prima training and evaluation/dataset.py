import torch
import random
import time
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import nibabel as nib
import sys,os,json,csv
from tqdm import tqdm

# collate function for ProtoDataset for clip training
def collate(max_tokens, patchify, device1, textpadtokenid, puttodevice = False):
    """
    max_tokens: max text token allowed
    patchify: the patchification module, used to convert raw old vqvae outputs (for entire slices in series) into "cube" tokens. For Rachel tokens this is used to add positional embeddings.
    device1: device to be put to, only if puttodevice=True
    textpadtokenid: the token id used to padding text
    puttodevice: whether to put the data to device
    """
    def ret(datas):
        device = device1

        textmax = 0 # maximum text length across batch
        study_imgmax = 0 # maximum length of study. 
        serie_imgmaxs = [0]*1000 # maximum token length of each serie location across batch. i.e. serie_imgmaxs[0] is the maximum token length across the FIRST series of each study in the batch, and so on

        datasnew = []
        study_lens = [] # number of series in each study
        serienames =[]
        hashnames = []
        studydescriptions = []
        for d in datas:
            hashnames.append(d['hash'])
            serienames.append(d['serienames'])
            studydescriptions.append(d['studydescription'])
            if 'coordinates' in d:
                coords = d['coordinates']
            else:
                coords = None
            datanew = [patchify(d['visual'],coords=coords),d['text'],d['textlen']] # patchify cuts up each flat representation into cubes for old tokens. For Rachel tokens it does nothing.

            visuals,text,textlen = datanew
            datasnew.append(datanew)
            if textlen > textmax:
                textmax = textlen
            for i,im in enumerate(visuals):
                if len(im) > serie_imgmaxs[i]:
                    serie_imgmaxs[i] = len(im)
            study_lens.append(len(visuals))
            if len(visuals) > study_imgmax:
                study_imgmax = len(visuals)
        
        serienames = convert_serienames_to_tensor(serienames)
        study_lens = torch.LongTensor(study_lens)
        visuals = [] # this variable is used to store the batch-processed visuals
        visuals = []
        for i in range(study_imgmax):
            visuals.append([])
        serie_lenss = [] # how long is each serie, with shape study_imagemax by batch_size. To find the length of ith serie in jth study in the batch, it is stored in serie_lenss[i][j]
        for i in range(study_imgmax):
            serie_lenss.append([])

        texts = []
        textlens = [] # length of texts
        for visual,text,textlen in datasnew:
            for i,im in enumerate(visual): # for each serie
                sizes = list(im.shape)
                h=sizes[0]
                imgpadlen = serie_imgmaxs[i] - h
                sizes[0] = imgpadlen
                imgpad = torch.zeros(sizes) # pad the serie sequence of tokens to max length
                visuals[i].append(torch.cat([im,imgpad],dim=0))
                serie_lenss[i].append(len(im))
            for i in range(len(visual),study_imgmax):
                # Pad along number-of-series dimension (i.e. if the current study has less series than the max number of series, pad a few empty ones)
                sizes = list(visual[0].shape)
                sizes[0] = serie_imgmaxs[i]
                visuals[i].append(torch.zeros(sizes))
                #visuals[i].append(torch.zeros(serie_imgmaxs[i],visual[0].size()[-1]))
                serie_lenss[i].append(0)
            
            # text sequence processing (padding)
            textpadlen = textmax - textlen
            textpad = torch.full((textpadlen,),textpadtokenid)
            texts.append(torch.cat([text,textpad],dim=0))
            textlens.append(textlen)
        

        
        # collate study description
        sdmaxlen = 0
        for tensor in studydescriptions:
            if len(tensor) > sdmaxlen:
                sdmaxlen = len(tensor)
        studydesc = torch.zeros(len(studydescriptions),sdmaxlen)
        for i,tensor in enumerate(studydescriptions):
            studydesc[i,0:len(tensor)] = tensor

        if device != 'cpu' and not puttodevice:
            device='cpu'
        retdict = {'text':torch.stack(texts,dim=0).long().to(device),'textlen':torch.LongTensor(textlens).to(device),'serienames':serienames.to(device),'hash':hashnames,'studydescription':studydesc}

        serie_lenss = torch.LongTensor(serie_lenss).transpose(0,1).to(device) # transpose the tensor so that the batch_size is first dimension, so DataParallel does not mess up
        retdict['visual'] = [torch.stack(ims,dim=0).to(device) for ims in visuals]
        retdict['lens'] = study_lens.to(device)
        retdict['lenss'] = serie_lenss
        return retdict
    return ret

        



# collate function for visual and hash only. Used for batched encoding for frozen CLIP classification if uselabels is false; Used for end-to-end classification training if uselabels is True
def collatevisualhash(patchify, device, uselabels=False,puttodevice=False):
    """
    patchify: the patchification module, used to convert raw old vqvae outputs (for entire slices in series) into "cube" tokens. For Rachel tokens, this module does nothing.
    device: generally not used (unless puttodevice is set to true)
    uselabels: whether to include classification labels
    puttodevice: set to true if you want to put things on device. Currently only supports usevarlen + separateseries + no orientation
    """
    def ret(datas):
        study_imgmax = 0 # maximum length of study. 
        serie_imgmaxs = [0]*1000 # maximum token length of each serie location across batch. i.e. serie_imgmaxs[0] is the maximum token length across the FIRST series of each study in the batch, and so on

        datasnew = []
        study_lens = [] # number of series in each study
        serienames = []
        labels = []
        studydescriptions = []
        for d in datas:
            serienames.append(d['serienames'])
            studydescriptions.append(d['studydescription'])
            if 'coordinates' in d:
                coords = d['coordinates']
            else:
                coords = None
            patched = patchify(d['visual'],coords = coords)
            label = d['label'] if uselabels else None
            datanew = [patched,label,d['hash']]
            datasnew.append(datanew)
            study_lens.append(len(patched))
            visual,_,s = datanew
            if len(visual) > study_imgmax:
                study_imgmax = len(visual)
            for i,im in enumerate(visual):
                if len(im) > serie_imgmaxs[i]:
                    serie_imgmaxs[i] = len(im)    
        
        study_lens = torch.LongTensor(study_lens)
        visuals = []
        strs = []
        visuals = []
        for i in range(study_imgmax):
            visuals.append([])
        serie_lenss = [] # how long is each serie, with shape study_imagemax by batch_size. To find the length of ith serie in jth study in the batch, it is stored in serie_lenss[i][j]
        for i in range(study_imgmax):
            serie_lenss.append([])
        for visual,label,s in datasnew:
            for i,im in enumerate(visual): # for each serie
                sizes = list(im.shape)
                h=sizes[0]
                imgpadlen = serie_imgmaxs[i] - h
                sizes[0] = imgpadlen
                imgpad = torch.zeros(sizes) # pad the serie sequence of tokens to max length
                visuals[i].append(torch.cat([im,imgpad],dim=0))
                serie_lenss[i].append(len(im))
            for i in range(len(visual),study_imgmax):
                # Pad along number-of-series dimension (i.e. if the current study has less series than the max number of series, pad a few empty ones)
                sizes = list(visual[0].shape)
                sizes[0] = serie_imgmaxs[i]
                visuals[i].append(torch.zeros(sizes))
                #visuals[i].append(torch.zeros(serie_imgmaxs[i],visual[0].size()[-1]))
                serie_lenss[i].append(0)
            
            strs.append(s)
            labels.append(label)
        serienames = convert_serienames_to_tensor(serienames)

        # collate study description
        sdmaxlen = 0
        for tensor in studydescriptions:
            if len(tensor) > sdmaxlen:
                sdmaxlen = len(tensor)
        studydesc = torch.zeros(len(studydescriptions),sdmaxlen)
        for i,tensor in enumerate(studydescriptions):
            studydesc[i,0:len(tensor)] = tensor
        
        if uselabels:
            if isinstance(labels[0],torch.Tensor):
                labels = torch.stack(labels).long()
            else:
                labels = torch.LongTensor(labels)

        serie_lenss = torch.LongTensor(serie_lenss).transpose(0,1)
        if puttodevice:
            visuals = [[im.to(device) for im in ims] for ims in visuals]
            study_lens = study_lens.to(device)
            serie_lenss = serie_lenss.to(device)
            serienames = serienames.to(device)
            studydesc = studydesc.to(device)
        return {'visual':[torch.stack(ims,dim=0) for ims in visuals],'lens':study_lens,'lenss':serie_lenss,'hash':strs,'serienames':serienames,'labels':labels,'studydescription':studydesc}
    return ret


# collate function for seriename dataset
def collateserienameclip(patchifier):
    def ret(xs):
        rawseries = [x[0] for x in xs]
        coords = [x[1] for x in xs]
        hashseries = [x[3] for x in xs]
        series = patchifier(rawseries,coords)
        lens = torch.LongTensor([len(ser) for ser in series])
        maxlen = lens.max()
        visual = torch.zeros(len(xs),maxlen,series[0].size()[-1])
        for i,s in enumerate(series):
            visual[i][0:lens[i]] = s
        serienames = [x[2] for x in xs]
        serienames = convert_serienames_to_tensor([serienames])[0]
        return {'visual':visual,'lens':lens,'serienames':serienames,'hash+series':hashseries}
    return ret


# This is basically a collate function for serienames
def convert_serienames_to_tensor(serienames):
    tensors = []
    max1 = 0
    max2 = 0
    for b1 in serienames:
        if len(b1) > max1:
            max1 = len(b1)
        for b2 in b1:
            if len(b2) > max2:
                max2 = len(b2)
    ret = torch.zeros(len(serienames),max1,max2).long()
    for i in range(len(serienames)):
        for j in range(len(serienames[i])):
            t = serienames[i][j]
            ret[i][j][0:len(t)] = t
    return ret


# A simple collate function for 3-tuples. Used for frozen CLIP classification training
def collateembhash(lis):
    ts = []
    ls = []
    hs = []
    for t,l,h in lis:
        ts.append(t)
        ls.append(l)
        hs.append(h)
    return torch.stack(ts,dim=0),torch.LongTensor(ls),hs



# sorting key function for prioritizing which series to include.
def keyfunc(aa):
    # aa: aa[0] is size of series, aa[2] is name of series
    extra = 0
    if 'unk' in aa[2].lower():  # we prioritize serie names without unk
        extra = 100000
    return aa[0] + extra

# helper function for getting total number of tokens of all series in a study
def gettotal(a):
    total = 0
    for l in a:
        total += l[0]
    return total


# method for preprocessing full reports
def preprocesstext(text,aug=False,splitfinding = False):
    # get rid of text before useful information in a report
    if splitfinding:
        texts = text.split('FINDINGS:')
        if len(texts) >= 2:
            text = text[text.index('FINDINGS:'):]
        else:
            texts = text.split('Findings:')
            if len(texts) >= 2:
                text = text[text.index('Findings:'):]
            else:
                texts = text.split('INTERPRETATION:')
                if len(texts) >= 2:
                    text = text[text.index('INTERPRETATION:'):]
    # get rid of text about who dictated the report
    texts = text.split('Dictated by:')
    if len(texts) >= 2:
        c = text.rindex('Dictated by:')
        return text[0:c]
    return text

# method for preprocessing shortened (summarized) reports
def preprocessshortenedtext(text,textlimit,tokenizer,istrain):
    # split the report into list items
    items = []
    lines = text.split('\n')
    for line in lines:
        if len(line) < 3:
            continue
        if '. ' not in line:
            items.append(line)
        else:
            items.append(line[line.index('. ')+2:])


    # shuffle items if traininig
    if istrain:
        random.shuffle(items)
    # remove items if the shortened report is longer than text length limit
    while True:
        ret = ''
        for i,item in enumerate(items):
            ret += str(i+1)+'. '+item+'\n'
        tok = tokenizer(ret)
        if len(tok) < textlimit:
            break
        items = items[:-1]
    return ret[:-1]


from transformers import AutoTokenizer, GPT2Tokenizer
import random


# abstract dataset class
class ProtoDataset(torch.utils.data.Dataset):
    def __init__(self, datajson, datarootdir, text_max_len, is_train, tokenizer, vqvaename, pt_limit = 11, series_dropout_rate = 0.0, split_finding_rate = 0.0, val_size=254,include_hash=False,visualhashonly=False,forcereportfromcsv=None,percentage=None,upsample_abnormal=0,novisualaug=False, tokendropout = None,seriename_dropout = None, prospectivedatalist = None, prospective=False, exclude_series=[],nosplit=False, notextdownstream = False, embname = 'emb', forceunkseriename=False,serienumcheck=True):

        """
        datajson: path to dataset json file (such as "glmv7.json")
        datarootdir: root directory to where the embeddings are stored
        text_max_len: max length for text input, in tokens
        is_train: train dataset or val dataset
        tokenizer: the text tokenizer to use
        vqvaename: name of vqvae used to generate the tokens
        pt_limit: not used. ignore
        series_dropout_rate: chance of dropping out an entire serie
        split_finding_rate: chance of split finding in full reports
        val_size: size of validation set
        include_hash: whether to include hash in output. I think we do it regardless of this variable
        visualhashonly: whether to only output visual and hash
        forcereportfromcsv: whether to replace the default full reports with reports from a csv
        percentage: otsu cutoff percentage
        upsample_abnormal: whether to upsample abnormals. The number indicate the number of times of copying the abnormal data.
        novisualaug: if this is set to true, no augmentation is applied to visual
        tokendropout: the rate of using token dropout. if is none, no token dropout.
        seriename_dropout: the rate of changing seriename to 'unk' as an augmentation. if none, no seriename dropout
        prospectivedatalist: data list csv used for constructing the initial prospective dataset
        prospective: whether this is the prospective test set
        exclude_series: serie names to exclude
        nosplit: if set to true, do not remove validation set
        notextdownstream: whether this dataset is a no-text-downstream task
        embname: the name of the 'emb' folder
        forceunkseriename: make all serienames "unk" (so basically inferencing with no seriename knowledge)
        serienumcheck: whether to enforce at least 2 series per study
        """

        # load alternate report csv
        self.reportcsvdict = None
        if forcereportfromcsv is not None:
            self.reportcsvdict = {}
            reader = csv.reader(open(forcereportfromcsv))
            for row in reader:
                if len(row) >= 2:
                    self.reportcsvdict[row[0]] = row[1]
                else:
                    print('warning: unreadable row in reportcsv: '+str(row))
        # load datajson
        self.datas = json.load(open(datajson))

        # split train set and val set
        if (not prospective) and (not nosplit):
            if is_train:
                self.datas = self.datas[0:-val_size]
            else:
                self.datas = self.datas[-val_size:]
        
        # upsample abnormals
        if upsample_abnormal > 0 and is_train:
            from abnormaltextfilter import getabnormallist
            abnormallist = set(getabnormallist(forcereportfromcsv))
            newdatas = []
            for data in self.datas:
                studypath,_,_,_  = data
                hashname = studypath.split('/')[-1]
                newdatas.append(data)
                if hashname in abnormallist:
                    for _ in range(upsample_abnormal):
                        newdatas.append(data)
            self.datas = newdatas

        # set some variables
        self.percentage=percentage # percentage of otsu cutoff
        self.is_train = is_train # train or val
        self.datalen = len(self.datas) # length of data
        self.text_max_len = text_max_len # text length limit
        self.pt_limit = pt_limit # not used. ignore
        self.novisualaug = novisualaug # not use visual aug if set to true
        self.tokendropout = tokendropout # token dropout rate
        self.seriename_dropout = seriename_dropout # chance of dropping serie name
        self.exclude_series = exclude_series # serie names that needs to be excluded
        self.embname = embname # the name of the emb folder
        self.forceunkseriename = forceunkseriename # inference with all serie names as "unk"

        # get tokenizer for language model
        if tokenizer == 'biomed':
            self.tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        elif tokenizer == 'tinyllama':
            self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        elif tokenizer == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            raise NotImplementedError

        # set some more variables
        self.vqvaename = vqvaename # name of vqvae used for the tokens
        self.eos_id = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0] # eos token for tokenizer
        self.seriedropout = series_dropout_rate # series drop out rate
        self.splitfindingrate = split_finding_rate # chance of split / unsplit findings in reports
        self.include_hash=include_hash # whether to include hash in output (I think we do it regardless of this variable now)
        self.visualhashonly=visualhashonly # whether to only return visual and hash
        
    def __len__(self):
        return self.datalen
    def gethash(self,idx):
        return self.datas[idx][0].split('/')[-1]
    def __getitem__(self,idx):
        out = self.getitem(idx)
        return out

    def getitem(self,idx):
        raise NotImplementedError

    # only getting text
    def get_text(self,idx):
        # if using default reports
        if self.reportcsvdict is None:
            _,_,report,_ = tuple(self.datas[idx])
            return preprocesstext(report,splitfinding=True)
        else: # if using replaced reports
            studypath,_,report,_ = tuple(self.datas[idx])
            hashname = studypath.split('/')[-1]
            if hashname not in self.reportcsvdict: # if no reports found
                print('Warning: '+hashname+' has default report but no report from csv')
                report = preprocesstext(report,splitfinding=True)
            else:
                report = self.reportcsvdict[hashname]
            report = preprocessshortenedtext(report,self.text_max_len,self.tokenizer,self.is_train,text_keyword_filter = self.text_keyword_filter)
            return report

    # only getting text in dict format
    def get_text_dict(self,report,splitfinding):
        # process text
        if self.reportcsvdict is None:
            report = preprocesstext(report,splitfinding=splitfinding)
        else:
            report = preprocessshortenedtext(report,self.text_max_len,self.tokenizer,self.is_train,text_keyword_filter = self.text_keyword_filter)
        text = self.tokenizer(report+'<|endoftext|>')['input_ids']
        if len(text) > self.text_max_len:
            text = text[0:self.text_max_len]
            text[-1] = self.eos_id
            textlen = self.text_max_len
        else:
            textlen = len(text)
        return {'text':torch.LongTensor(text),'textlen':textlen}

    # onlt getting the path to the study
    def get_path(self,idx):
        studypath,_,_,_ = tuple(self.datas[idx])
        return studypath

    # find data by hash
    def find_by_hash(self,hash, get_id_only=False):
        for i,d in enumerate(self.datas):
            assert isinstance(d[0],str)
            if hash == d[0].split('/')[-1]:
                if get_id_only:
                    return i
                return self.getitem(i)

    

# this function converts the serie name text to LongTensor
char_to_index = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz_+-0123456789.*(),")}
def chartovec(s):
    ret = []
    for c in s.lower():
        try:
            ret.append(char_to_index[c]+1)
        except:
            ret.append(45)
    ret.append(46)
    return torch.LongTensor(ret)

# Dataset for new tokens from Rachel as well as Token256
class RachelDataset(ProtoDataset):
    def getitem(self,idx,custom_input = None):
        if custom_input is not None:
            studypath,series,report,studydescr = custom_input
        else:
            studypath,series,report,studydescr = tuple(self.datas[idx])
        hashname = studypath.split('/')[-1]

        # split finding if necessary
        splitfinding=False
        if random.random() < self.splitfindingrate:
            splitfinding=True

        # replace report if necessary
        if self.reportcsvdict is not None:
            if hashname not in self.reportcsvdict:
                print('Warning: '+hashname+' has default report but no report from csv')
                report = preprocesstext(report,splitfinding=splitfinding)
            else:
                report = self.reportcsvdict[hashname]
        
        
        # if only getting textdict:
        if hasattr(self,'textdictonly') and self.textdictonly:
            return self.get_text_dict(report,splitfinding)


        out_series = []

        assert len(series) > 0
        attempts = 0
        while len(out_series) == 0: # we repeat many times to ensure that we don't drop-out all series
            if attempts >= 5: # if fails over 10 times, we replace with random other study.
                print(hashname+' cannot get enough good series!')
                if self.is_train:
                    return self.getitem(random.randint(0,len(self)-1))
                else:
                    return self.getitem(idx-1)
            attempts += 1

            # process all series
            out_series = []
            for serie,_ in series:
                if serie in self.exclude_series:
                    continue

                if self.is_train and (not self.novisualaug) and self.seriedropout > 0: # series dropout
                    if random.random() < self.seriedropout:
                        continue
                seriepath = studypath + '/' + serie + '/'+self.embname+'/'+ self.vqvaename
                allembs = torch.load(seriepath+'/stacked/stacked.pt',map_location='cpu') # load embeddings
                percentagemeta = json.load(open(seriepath+'/emb_meta.json')) # load emb meta
                percentageadjust = 0
                if self.is_train and (not self.novisualaug):
                    percentageadjust = random.randint(-2,2)
                for percent in range(self.percentage+percentageadjust, -1, -1): # try lower percentages if high percentage has small number of tokens
                    embs,embspos,posmap = filtercoords(percentagemeta,percent,allembs) # filter embedding and coordinates according to ostu percentage
                    if len(embspos) > 25 and percent > 0: # if we have enough tokens at a positive percentage, we're good
                        break
                    if percent == 0: # if we reached 0%
                        if len(embspos) > 0: # if all tokens are 0%, we ignore this serie
                            embspos = []
                        else: # if no token has otsu percentage, we use all tokens (according to Asadur)
                            embs = allembs
                            embspos = []
                            if len(embs) > 5000 or len(embs) == 0: # if all tokens is too big or too small
                                break
                            for i in range(len(embs)):
                                embspos.append(percentagemeta['emb_index'][str(i)]) # use all coordinates
                            embspos = torch.LongTensor(embspos)
            
                if len(embspos) == 0: # ignore serie if no embedding
                    continue
                if torch.isnan(embs).any(): # nan check
                    print('wtf why man')
                    continue
                if self.is_train and self.tokendropout is not None and (not self.novisualaug):
                    # replacing old inefficient dropout sampling with new efficient ones
                    a = torch.ones(len(embs)) * (1 - self.tokendropout)
                    mask = torch.bernoulli(a)
                    indexes = mask.nonzero().squeeze()
                    embs = embs[indexes]
                    embspos = embspos[indexes]
                    posmap = posmap[indexes]
                out_series.append((embs,embspos,serie,None,posmap))

        # randomly shuffle series if training        
        if self.is_train and (not self.novisualaug):
            random.shuffle(out_series)
        serienamestr = [o[2] for o in out_series] # serie names in string
        serienames = [] # serie names vectorized
        for o in out_series:
            if (self.seriename_dropout is not None and self.is_train and (not self.novisualaug) and random.random() < self.seriename_dropout) or self.forceunkseriename:
                serienames.append(chartovec('unk'))
            else:
                serienames.append(chartovec(o[2]))
        out_series_pos = [o[1] for o in out_series] # embedding coordinates
        posmaps = [o[4] for o in out_series]
        out_series = [o[0] for o in out_series] # embeddings
        

        studydescr = chartovec(studydescr)
        if self.visualhashonly: # only return visual and hash
            return {'visual':out_series,'hash':studypath.split('/')[-1],'serienames':serienames,'coordinates':out_series_pos,'studydescription': studydescr,'posmap':posmaps,'serienamestr':serienamestr}
        
        # replace report if necessary
        if self.reportcsvdict is None: 
            report = preprocesstext(report,splitfinding=splitfinding)
        else:
            report = preprocessshortenedtext(report,self.text_max_len,self.tokenizer,self.is_train)

        # process text
        text = self.tokenizer(report+'<|endoftext|>')['input_ids']
        if len(text) > self.text_max_len:
            text = text[0:self.text_max_len]
            text[-1] = self.eos_id
            textlen = self.text_max_len
        else:
            textlen = len(text)
        
        ret = {'visual':out_series,'text':torch.LongTensor(text),'textlen':textlen,'hash':hashname,'serienames':serienames,'coordinates':out_series_pos,'studydescription': studydescr,'posmap':posmaps,'serienamestr':serienamestr}
        
        return ret

# helper function for filtering coordinates based on pixel intensity
def filtercoords(meta,percentagetouse,embs,fillhole=True, debuginfo='None'):
    percentagemeta = meta['OtsuThresholds']
    uses = []
    if fillhole:
        metadict = {}
        for idx in meta['emb_index']:
            x,y,z = meta['emb_index'][idx]
            metadict[x*1000000+y*1000+z] = int(idx)
    for i in range(percentagetouse,101):
        uses += percentagemeta[str(i)]['OutfillCoords']
        if fillhole and i <= 20:
            infillcoords = percentagemeta[str(i)]['InfillCoords']
            uses += [(metadict[b[0]*1000000+b[1]*1000+b[2]],b) for b in infillcoords]
    useids = torch.LongTensor([u[0] for u in uses]) # the embs to use
    embspos = torch.LongTensor([u[1] for u in uses]) # the coordinates of the embs
    return embs[useids],embspos,useids



# a subdataset of a dataset, with randomly selected sizes
class SubDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,limit):
        # dataset: the original dataset
        # limit: the size limit we want
        self.fulllen = len(dataset)
        self.dataset = dataset
        self.ids = random.sample(range(self.fulllen),limit) # subsample the dataset
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,idx):
        return self.dataset[self.ids[idx]]
    def resample(self):
        self.ids = random.sample(range(self.fulllen),len(self.ids)) # resubsample the dataset
    def getratio(self): # this function should only be used during BCE-based classification, not during CLIP training
        labels = 0
        for i in range(len(self)):
            labels += self.dataset.getlabels(self.ids[i])
        neglabels = len(self) - labels
        return neglabels / labels

# dataset used for training clip between series and serienames
class SerieNameCLIPDataset(torch.utils.data.Dataset):
    def __init__(self,datajson,is_train,vqvaename,tokendropout=0,novisualaug=False,percentage=5,val_size=254,special_book=None,nosplit = False):
        origjson = json.load(open(datajson))
        if nosplit: 
            myorigjson = origjson
        else:
            if is_train:
                myorigjson = origjson[0:-val_size]
            else:
                myorigjson = origjson[-val_size:]

        toincludenamedict = {d[0]:d[1] for d in special_book} if special_book is not None else None

        self.datas = []
        for data in myorigjson:
            if toincludenamedict is not None:
                h = data[0].split('/')[-1]
                if h not in toincludenamedict:
                    continue
                self.datas.append((data[0],toincludenamedict[h]))
            else:
                for serie in data[1]:
                    self.datas.append((data[0],serie))
        
        self.is_train = is_train
        self.tokendropout = tokendropout
        self.novisualaug = novisualaug
        self.vqvaename = vqvaename
        self.percentage = percentage
    def __len__(self):
        return len(self.datas)
    def __getitem__(self,idx):
        studypath,serieo = self.datas[idx]
        if isinstance(serieo,str):
            serie = serieo
            orientation = None
        else:
            serie,orientation = serieo
        seriepath = studypath + '/' + serie + '/emb/'+ self.vqvaename
        allembs = torch.load(seriepath+'/stacked/stacked.pt',map_location='cpu') # load embeddings
        percentagemeta = json.load(open(seriepath+'/emb_meta.json')) # load emb meta
        percentageadjust = 0
        if self.is_train and (not self.novisualaug):
            percentageadjust = random.randint(-2,2)
        for percent in range(self.percentage+percentageadjust, -1, -1): # try lower percentages if high percentage has small number of tokens
            embs,embspos,posmap = filtercoords(percentagemeta,percent,allembs) # filter embedding and coordinates according to ostu percentage
            if len(embspos) > 25 and percent > 0: # if we have enough tokens at a positive percentage, we're good
                break
            if percent == 0: # if we reached 0%
                if len(embspos) > 0: # if all tokens are 0%, we ignore this serie
                    embspos = []
                else: # if no token has otsu percentage, we use all tokens (according to Asadur)
                    embs = allembs
                    embspos = []
                    if len(embs) > 5000 or len(embs) == 0: # if all tokens is too big or too small
                        break
                    for i in range(len(embs)):
                        embspos.append(percentagemeta['emb_index'][str(i)]) # use all coordinates
                    embspos = torch.LongTensor(embspos)
        
        if len(embspos) == 0: # ignore serie if no embedding
            return self[random.randint(0,len(self)-1)]
        if self.is_train and self.tokendropout is not None and (not self.novisualaug): # perform token dropout
            newembs = []
            newembspos = []
            newposmap = []
            for i in range(len(embspos)):
                if random.random() >= self.tokendropout:
                    newembs.append(embs[i])
                    newembspos.append(embspos[i])
                    newposmap.append(posmap[i])
            embs = torch.stack(newembs)
            embspos = torch.stack(newembspos)
            posmap = torch.stack(newposmap)
        hashseries = studypath.split('/')[-1]+'|'+serie
        return embs,embspos,chartovec(serie),hashseries,orientation,posmap
    




