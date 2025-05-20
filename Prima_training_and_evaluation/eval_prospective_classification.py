import yaml, argparse, json, csv, copy
from tqdm import tqdm
from dataset import MrDataset, collatevisualhash
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


# compute balanced accuracy score
def evalacc(gt, pred, offset):
    preds = [((p - offset) >= 0) for p in pred]
    tn, fp, fn, tp = confusion_matrix(gt, preds).ravel()
    return balanced_accuracy_score(gt, preds), tp / (tp + fn), tn / (tn + fp)


# compute auc-roc
def evalauc(gt, pred):
    auc = roc_auc_score(gt, pred)
    return auc


if __name__ == '__main__':
    cf_fd = parse_args()
    config = yaml.load(cf_fd, Loader=yaml.FullLoader)
    thedataset = MrDataset
    # prepare prospective dataset for inference
    dataset = thedataset(datajson=config['data']['datajson'],
                         datarootdir=config['data']['datarootdir'],
                         tokenizer='biomed',
                         text_max_len=200,
                         is_train=False,
                         vqvaename=config['data']['vqvaename'],
                         percentage=config['data']['percentage'],
                         prospective=True,
                         nosplit=True,
                         visualhashonly=True)

    # prepare inference model
    model = torch.load(config['eval']['ckpt'], map_location='cpu')
    p = model.module.patchifier.cpu()
    model = torch.nn.DataParallel(model.module.visual_model,
                                  device_ids=[0]).to('cuda:0')
    evaldict = json.load(open(config['eval']['evalset']))

    # load classification heads and prospective labels
    heads = {}
    poslists = {}
    for name in evaldict:
        modellists = evaldict[name][1]
        for m, _ in modellists:
            heads[m] = torch.load(m, map_location='cpu')
        poslist = open(evaldict[name][0]).read().split('\n')
        poslists[name] = poslist

    # record results
    preddict = {
        name: {m: []
               for m, _ in evaldict[name][1]}
        for name in evaldict
    }
    gts = {name: [] for name in evaldict}
    model.module.patdis = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        collate_fn=collatevisualhash(p, 'cpu'),
        shuffle=False,
        num_workers=7,
        timeout=7200)
    with torch.no_grad():

        # run inference
        embs = []
        imgencs = []
        for batch in tqdm(dataloader):
            # run clip backbone inference
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                embeds = model(xdict=batch, retpool=True)
                embs.append(embeds)
                for h in batch['hash']:
                    for name in evaldict:
                        if h in poslists[name]:
                            gts[name].append(True)
                        else:
                            gts[name].append(False)
            # run head inference
            for name in evaldict:
                modellist = evaldict[name][1]
                for m, i in modellist:
                    #with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                    pred = heads[m](
                        embeds.to('cpu').float())[:, i] - heads[m].thresh
                    preddict[name][m].append(pred)

        # run metrics
        for name in preddict:
            for m in preddict[name]:
                preddict[name][m] = torch.cat(preddict[name][m],
                                              dim=0).detach().cpu().tolist()
                bas, se, sp = evalacc(gts[name], preddict[name][m], 0)
                print(name + ' model:  balanced accuracy score: ' + str(bas) +
                      '    \t\t' + m)
                auc = evalauc(gts[name], preddict[name][m])
                print(name + ' model:  auc-roc: ' + str(auc) + '    \t\t' + m)
