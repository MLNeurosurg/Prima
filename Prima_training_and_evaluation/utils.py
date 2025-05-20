import torch


# get best threshold for highest accuracy
def getbestthresh(gts, preds):
    totals = len(gts)
    bestacc = 0.0
    bestthresh = -100
    for thresh in [c / 5 for c in range(-25, 11)]:
        corrects = 0
        for i in range(len(gts)):
            if gts[i] == 0 and preds[i] < thresh:
                corrects += 1
            elif gts[i] == 1 and preds[i] >= thresh:
                corrects += 1
        acc = corrects / totals
        if acc > bestacc:
            bestacc = acc
            bestthresh = thresh
    return bestthresh


# CLIPSCORE calculation
def clipscore(embs1, embs2):
    assert len(embs1) == len(embs2)
    outs = torch.zeros(len(embs1))
    for i in range(len(embs1)):
        outs[i] = torch.dot(embs1[i], embs2[i])
    return outs.mean().item() * 100


# evaluate top-k retrieval accuracy
def retrievaleval(embs1, embs2, k):
    sims = torch.matmul(embs1, embs2.t())

    textretrievals = torch.topk(sims, k=k, dim=0).indices
    imageretrievals = torch.topk(sims, k=k, dim=1).indices.transpose(0, 1)
    tcorrect = torch.sum(
        textretrievals == torch.arange(len(embs1)).to(embs1.device))
    icorrect = torch.sum(
        imageretrievals == torch.arange(len(embs1)).to(embs1.device))
    return tcorrect / len(embs1), icorrect / len(
        embs1), textretrievals, imageretrievals
