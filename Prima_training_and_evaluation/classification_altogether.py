import torch
import sys, os, csv

sys.path.append(os.getcwd())
from tqdm import tqdm
from sklearn import metrics
from dataset import collatevisualhash, collateembhash, MrDataset

from utils import getbestthresh
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ClassificationTask:

    def __init__(self,
                 dataset,
                 visualmodel,
                 poslist,
                 patchify,
                 protobatchsize=12,
                 vallist=None,
                 retpool=True):

        self.allembeds = []
        self.vallistnames = vallist

        self.vallist = [open(v).read().split('\n') for v in vallist]

        # obtain visual embeddings from pre-trained clip visual model
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=protobatchsize,
                                             shuffle=False,
                                             num_workers=12,
                                             collate_fn=collatevisualhash(
                                                 patchify, device))
        with torch.no_grad():
            for d in tqdm(loader):
                h = d['hash']
                with torch.amp.autocast(device_type='cuda',
                                        dtype=torch.float16):
                    embeds = visualmodel(xdict=d, retpool=retpool)
                labels = [[(1 if (hh in pl) else 0) for pl in poslist]
                          for hh in h]
                for i, (tensor, label) in enumerate(zip(embeds, labels)):
                    self.allembeds.append((tensor.float(), label, h[i]))

        # obtain label for each diagnosis
        for i in range(len(self.allembeds)):
            t, l, h = self.allembeds[i]
            self.allembeds[i] = (t, [(1 if h in pl else 0)
                                     for pl in poslist], h)

        # split into train set and val set
        self.trainembeds, self.valembeds = self.split(self.allembeds)

        print(len(self.valembeds))

        # compute BCE pos weights
        ttotals = len(self.trainembeds)
        ones = torch.zeros(len(poslist)).long()
        for _, l, _ in self.trainembeds:
            ones += torch.LongTensor(l)
        rest = ttotals - ones
        ratios = rest / ones
        print('train totals: ' + str(ttotals))
        print('train ones: ' + str(ones))
        print('ratio: ' + str(ratios))

        self.trainembedsbalanced = self.trainembeds
        self.criterionweighted = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor(ratios).to(device))
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.visembedlen = len(self.allembeds[0][0])

    def trainandval(self, model, optimizer, batch_size=200):
        trainloader = torch.utils.data.DataLoader(self.trainembedsbalanced,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=collateembhash)
        valloader = torch.utils.data.DataLoader(self.valembeds,
                                                batch_size=batch_size,
                                                collate_fn=collateembhash)
        totalloss = 0.0
        totals = 0
        classnum = len(self.vallist)

        # train for 1 epoch
        for t, l, h in tqdm(trainloader):
            t = t.to(device)
            t *= (torch.randn_like(t) * 0.01 + 1)
            l = l.to(device)
            optimizer.zero_grad()
            out = model(t)
            loss = self.criterionweighted(out, l.float())
            loss.backward()
            optimizer.step()
            totalloss += loss.item() * len(l)
            totals += len(l)

        # run validation set
        trainloss = totalloss / totals
        totalloss = 0.0
        totals = 0
        corrects = []
        fullpred = []
        fulllabels = []
        correctpos = []
        correctneg = []
        embedpreds = []
        valsetindexmap = [[] for i in range(classnum)]
        with torch.no_grad():
            for t, l, h in valloader:
                t = t.to(device)
                ll = l.to(device)
                out = model(t)
                loss = self.criterion(out, ll.float())
                totalloss += loss.item() * len(l)
                totals += len(l)
                for i in range(len(l)):
                    pred = (out[i] > 0).long().cpu()
                    correctpos.append(
                        torch.logical_and(pred == 1, l[i] == 1).long())
                    correctneg.append(
                        torch.logical_and(pred == 0, l[i] == 0).long())
                    corrects.append((pred == l[i]).long())
                    fullpred.append(out[i])
                    fulllabels.append(l[i])
                    embedpreds.append((t[i], pred))
                    for j in range(classnum):
                        if h[i] in self.vallist[j]:
                            valsetindexmap[j].append(len(fullpred) - 1)

        # calculate val metrics
        valloss = totalloss / totals
        fullpred = torch.stack(fullpred).detach().cpu().numpy()
        fulllabels = torch.stack(fulllabels).detach().cpu().numpy()
        valacc = []
        valauc = []
        correctposlist = []
        correctneglist = []
        bestthreshs = []
        for j in range(classnum):
            correctssum = 0
            correctpossum = 0
            correctnegsum = 0
            values = []
            gts = []
            for i in valsetindexmap[j]:
                correctssum += corrects[i][j]
                correctpossum += correctpos[i][j]
                correctnegsum += correctneg[i][j]
                values.append(fullpred[i][j])
                gts.append(fulllabels[i][j])
            valacc.append(correctssum / len(valsetindexmap[j]))
            valauc.append(metrics.roc_auc_score(gts, values))
            correctposlist.append(correctpossum)
            correctneglist.append(correctnegsum)
            bestthreshs.append(
                getbestthresh(gts, values)
            )  # get the best cutoff threshold for binary classification

        return trainloss, valloss, valacc, valauc, correctposlist, correctneglist, (
            fullpred > 0).astype(np.int32), bestthreshs

    def split(self, pairs):
        # split all embeddings into training set and validation set, for classification head training purposes only
        vals = []
        trains = []
        fullvallist = []
        valhashset = set([])
        for vl in self.vallist:
            fullvallist += vl
        fullvallistset = set(fullvallist)
        print(len(pairs))
        for t, l, h in pairs:
            if h in fullvallistset:
                if h not in valhashset:
                    vals.append((t, l, h))
                    valhashset.add(h)
                    fullvallistset.remove(h)
                else:
                    print('weird')
            else:
                trains.append((t, l, h))
        print(fullvallistset)
        return trains, vals


# create empty list
def emptylist(num):
    ret = []
    for i in range(num):
        ret.append(0)
    return ret


# load clip-trained vision model
def loadvismodel(path, devices):
    m = torch.load(path, map_location='cpu')
    model = torch.nn.DataParallel(m.module.visual_model,
                                  device_ids=devices).to(device)
    model.module.patdis = False
    return model, m.module.patchifier.to('cpu')


# main training loop for a specific checkpoint
def train(protodataset,
          vismodelpath,
          poslist,
          vallist=None,
          epochs=45,
          classnum=1,
          devices=[0],
          savesite='.',
          cnames=[]):
    vismodel, patchify = loadvismodel(vismodelpath, devices)
    cls = ClassificationTask(protodataset,
                             vismodel,
                             poslist,
                             patchify,
                             vallist=vallist)
    bestvauc = emptylist(classnum)
    bestvacc = emptylist(classnum)
    bestaucfpred = [[] for i in range(classnum)]
    bestaccfpred = [[] for i in range(classnum)]
    newmodel = torch.nn.Sequential(torch.nn.Linear(cls.visembedlen,
                                                   4000), torch.nn.ReLU(),
                                   torch.nn.Linear(4000, 1000),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(1000, classnum))
    newmodel = torch.nn.DataParallel(
        newmodel, device_ids=config['train']['devices']).to(device)
    optim = torch.optim.RMSprop(newmodel.parameters(), lr=0.00001)
    for e in range(epochs):
        tloss, vloss, vacc, vauc, cpos, cneg, fpred, threshs = cls.trainandval(
            newmodel, optim)
        print('epoch ' + str(e) + ' train loss: ' + str(tloss))
        print('epoch ' + str(e) + ' val loss: ' + str(vloss))
        print('epoch ' + str(e) + ' val acc: ' + str(vacc))
        print('epoch ' + str(e) + ' val auc: ' + str(vauc))
        print('epoch ' + str(e) + ' val correct positive: ' + str(cpos))
        print('epoch ' + str(e) + ' val correct negative: ' + str(cneg))

        # save best checkpoints for each task
        for i in range(classnum):
            if vauc[i] > bestvauc[i]:
                bestvauc[i] = vauc[i]
                bestaucfpred[i] = fpred[:, i]
                newmodel.module.thresh = threshs[i]
                torch.save(newmodel.module,
                           savesite + '/bestauc_' + cnames[i] + '.pt')
    return bestvauc, bestvacc, bestaucfpred, bestaccfpred, cls


import yaml, argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


if __name__ == '__main__':
    cf_fd = parse_args()
    config = yaml.load(cf_fd, Loader=yaml.FullLoader)
    thedataset = MrDataset
    protodataset = thedataset(datajson=config['data']['datajson'],
                              datarootdir=config['data']['datarootdir'],
                              tokenizer='biomed',
                              text_max_len=200,
                              is_train=True,
                              vqvaename=config['data']['vqvaename'],
                              visualhashonly=True,
                              percentage=config['data']['percentage'],
                              novisualaug=True,
                              nosplit=config['data']['nosplit']
                              if 'nosplit' in config['data'] else False)
    nums = config['train']['nums']
    paths = [(config['train']['ckptsavedir'] + '/' + str(num) + '.pt', num)
             for num in nums]
    poslist = [
        open(pl).read().split('\n') for pl in config['cset']['txtnames']
    ]
    vallists = config['cset']['vallists']
    bestaucs = []
    bestaccs = []
    mysavesite = config['cset']['savesite']
    os.system('mkdir ' + mysavesite + '/' + config['cset']['markdate'])

    # run classification head training and validation for each included checkpoint
    for path, num in paths:
        bestvauc, bestvacc, bestaucfpred, bestaccfpred, cls = train(
            protodataset,
            path,
            poslist,
            vallist=vallists,
            epochs=config['train']['epochs'],
            classnum=config['cset']['classnum'],
            savesite=mysavesite + '/' + config['cset']['markdate'],
            cnames=config['cset']['names'],
            devices=config['train']['devices'])
        print(path + ' best val auc: ' + str(bestvauc))
        bestaucs.append(bestvauc)
        bestaccs.append(bestvacc)
    print('best auc and acc:')

    i = 0
    for auc, acc in zip(bestaucs, bestaccs):
        print(config['cset']['names'][i] + ' auc and acc:')
        print(str(auc) + ' ' + str(acc))
        i += 1
