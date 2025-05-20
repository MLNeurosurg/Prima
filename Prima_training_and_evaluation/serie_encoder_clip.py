import sys, os

sys.path.insert(0, os.getcwd())
import argparse
from dataset import collateserienameclip, SerieNameCLIPDataset, SubDataset
from model import SerieCLIP
from model_parts import clip_objective
from tqdm import tqdm
import yaml, time
import torch
import copy
import traceback

device = 'cuda:0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


def main(config):
    # choose dataset
    thedataset = SerieNameCLIPDataset

    # set otsu percentage (if using it)
    if 'percentage' not in config['data']:
        config['data']['percentage'] = None

    # create train and val datasets
    train_dataset = thedataset(datajson=config['data']['datajson'],
                               is_train=True,
                               vqvaename=config['data']['vqvaename'],
                               percentage=config['data']['percentage'],
                               tokendropout=config['train']['token_dropout'])
    val_dataset = thedataset(
        datajson=config['data']['datajson'],
        is_train=False,
        vqvaename=config['data']['vqvaename'],
        percentage=config['data']['percentage'],
    )

    # create checkpoint and logging directories
    os.system('mkdir ' + config['train']['ckptsavedir'])

    # create new model or load existing model
    model = SerieCLIP(config).to(device)
    patching = copy.deepcopy(model.patchifier.to('cpu'))
    model.patchifier.to(device)
    model = torch.nn.DataParallel(model, device_ids=config['train']['devices'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['train']['learning_rate'],
                                 weight_decay=config['train']['weight_decay'])

    cumbatchnum = 1
    if 'early_stop_step' in config['train']:
        train_sub_dataset = SubDataset(
            train_dataset,
            config['train']['early_stop_step'] * config['data']['batch_size'] *
            cumbatchnum)  # a subdataset if we do early stop
    else:
        train_sub_dataset = train_dataset
    train_loader = torch.utils.data.DataLoader(
        train_sub_dataset,
        batch_size=config['data']['batch_size'],
        collate_fn=collateserienameclip(patching),
        shuffle=True,
        num_workers=config['train']['num_train_loader_workers'],
        timeout=7200)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        collate_fn=collateserienameclip(patching),
        shuffle=False,
        num_workers=config['train']['num_val_loader_workers'],
        timeout=1200)

    for e in range(config['train']['epochs']):

        totalloss = 0.0
        totals = 0
        taccs1, iaccs1, taccs5, iaccs5 = [], [], [], []

        # if using subdataset, we need to resample before beginning of training
        if 'early_stop_step' in config['train']:
            train_sub_dataset.resample()

        try:
            cumulator = []
            for batchnum, batch in enumerate(tqdm(train_loader)):
                cumulator.append(batch)
                if cumbatchnum > 1:
                    if len(
                            cumulator
                    ) < cumbatchnum:  # cumulate a few batches for processing together
                        continue

                model.train()
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda',
                                        dtype=torch.float16):
                    out1list, out2list = [], []
                    for batchz in cumulator:
                        out1, out2 = model(batchz)
                        out1list.append(out1)
                        out2list.append(out2)
                    out1 = torch.cat(out1list, dim=0)
                    out2 = torch.cat(out2list, dim=0)
                    loss, losstemp = clip_objective(out1, out2,
                                                    model.module.temperature)

                losstemp.backward()
                cumulator = []
                # gradient clipping
                if 'gradient_clip' in config['train']:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['train']['gradient_clip'])

                # deal with warmup
                if e == 0 and 'warmup' in config['train'] and batchnum < config[
                        'train']['warmup'] * cumbatchnum:
                    warmup = (batchnum + 1) / config['train']['warmup']
                    for g in optimizer.param_groups:
                        g['lr'] = config['train']['learning_rate'] * warmup

                optimizer.step()
                totalloss += loss.item()

                # calculate training metrics
                if len(out1) == config['data']['batch_size'] * cumbatchnum:
                    totals += 1
                    with torch.no_grad():
                        tacc, iacc, _, _ = retrievaleval(out1, out2, 1)
                        taccs1.append(tacc)
                        iaccs1.append(iacc)
                        tacc, iacc, _, _ = retrievaleval(out1, out2, 5)
                        taccs5.append(tacc)
                        iaccs5.append(iacc)

        except RuntimeError as e:  # if errored out, figure out why. If just dataloader killed unexpectedly, we can ignore
            print(e)
            time.sleep(100)
            traceback.print_exc()
            if 'early_stop_step' in config['train']:
                train_sub_dataset = SubDataset(
                    train_dataset, config['train']
                    ['early_stop_step'])  # a subdataset if we do early stop
            else:
                train_sub_dataset = train_dataset
            train_loader = torch.utils.data.DataLoader(
                train_sub_dataset,
                batch_size=config['data']['batch_size'],
                collate_fn=collateserienameclip(patching),
                shuffle=True,
                num_workers=config['train']['num_train_loader_workers'],
                timeout=7200)
            continue
        taccs1 = torch.FloatTensor(taccs1)
        iaccs1 = torch.FloatTensor(iaccs1)
        taccs5 = torch.FloatTensor(taccs5)
        iaccs5 = torch.FloatTensor(iaccs5)
        print('train set text retrieval top 1 accuracy: ' +
              str(torch.mean(taccs1).item()))
        print('train set image retrieval top 1 accuracy: ' +
              str(torch.mean(iaccs1).item()))
        print('train set text retrieval top 5 accuracy: ' +
              str(torch.mean(taccs5).item()))
        print('train set image retrieval top 5 accuracy: ' +
              str(torch.mean(iaccs5).item()))
        print('epoch ' + str(e) + ' train loss: ' + str(totalloss / totals))

        # validation
        with torch.no_grad():
            model.eval()
            embs1 = []
            embs2 = []
            try:
                for batch in tqdm(val_loader):

                    with torch.amp.autocast(device_type='cuda',
                                            dtype=torch.float16):
                        out1, out2 = model(batch)
                    embs1.append(out1)
                    embs2.append(out2)
            except RuntimeError as e:
                print(e)
                traceback.print_exc()
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=config['data']['batch_size'],
                    collate_fn=collateserienameclip(patching),
                    shuffle=False,
                    num_workers=config['train']['num_val_loader_workers'],
                    timeout=1200)
                continue
            # compute validation metrics and statistics over the entire val set
            embs1 = torch.cat(embs1, dim=0)
            embs2 = torch.cat(embs2, dim=0)
            loss, _ = clip_objective(embs1, embs2)
            print("Val scaled loss: " + str(loss.item()))
            tacc, iacc, _, _ = retrievaleval(embs1, embs2, 1)
            print('val set text retrieval top 1 accuracy: ' + str(tacc))
            print('val set image retrieval top 1 accuracy: ' + str(iacc))
            print("Val scaled loss: " + str(loss.item()))
            tacc5, iacc5, tr, ir = retrievaleval(embs1, embs2, 5)
            print('val set text retrieval top 5 accuracy: ' + str(tacc5))
            print('val set image retrieval top 5 accuracy: ' + str(iacc5))
            torch.save(model, config['train']['ckptsavedir'] + '/last.pt')
            torch.save(model,
                       config['train']['ckptsavedir'] + '/' + str(e) + '.pt')


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


if __name__ == '__main__':
    cf_fd = parse_args()
    config = yaml.load(cf_fd, Loader=yaml.FullLoader)
    main(config)
