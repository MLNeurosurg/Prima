import sys,os
sys.path.insert(0,os.getcwd())
import argparse
from dataset import collate, RachelDataset, SubDataset
from model import CLIP
from model_parts import clip_objective, patdis_objective
from tqdm import tqdm
import yaml,time
import torch
import copy
import traceback
from utils import clipscore, retrievaleval

device = 'cuda:0'
os.environ['TOKENIZERS_PARALLELISM']='false'
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
    thedataset = RachelDataset

    # set otsu percentage (if using it)
    if 'percentage' not in config['data']:
        config['data']['percentage'] = None
    
    # patient series discrimination
    patdis = None
    if 'patient_series_discrimination' in config['train']:
        patdis = config['train']['patient_series_discrimination']


    # create train and val datasets
    train_dataset = thedataset(datajson=config['data']['datajson'],
            datarootdir=config['data']['datarootdir'],
            tokenizer=config['data']['tokenizer'],
            text_max_len=config['data']['text_max_tokens'],
            is_train=True,
            vqvaename = config['data']['vqvaename'],
            series_dropout_rate = config['train']['series_dropout_rate'],
            split_finding_rate = config['train']['split_finding_rate'],
            forcereportfromcsv = config['data']['reportcsv'],
            percentage = config['data']['percentage'],
            upsample_abnormal = config['data']['upsample_abnormal'],
            tokendropout = config['train']['token_dropout'],
            seriename_dropout = config['train']['seriename_dropout'],
            prospectivedatalist = None if 'datalistcsv' not in config['data'] else config['data']['datalistcsv']
            )
    
    val_dataset = thedataset(datajson=config['data']['datajson'],
            datarootdir=config['data']['datarootdir'],
            tokenizer=config['data']['tokenizer'],
            text_max_len=config['data']['text_max_tokens'],
            is_train=False,
            vqvaename = config['data']['vqvaename'],
            forcereportfromcsv = config['data']['reportcsv'],
            percentage = config['data']['percentage']
            )
    

    # create checkpoint and logging directories
    os.system('mkdir '+config['train']['ckptsavedir'])

    # create new model or load existing model
    if 'clip_ckpt' not in config['model']:
        model = CLIP(config).to(device)
        patching = copy.deepcopy(model.patchifier.to('cpu'))
        model.patchifier.to(device)
        model = torch.nn.DataParallel(model,device_ids=config['train']['devices'])
    else:
        print('loading existing clip model')
        model = torch.load(config['model']['clip_ckpt'])
        if 'forcetextckpt' in config['model']['text'] and config['model']['text']['forcetextckpt']:
            model.module.replacetextmodel(config['model']['text']['ckpt_path'],config['model']['feature_dim'])
        patching = copy.deepcopy(model.module.patchifier.to('cpu'))
        model.module.patchifier.to(device)
        model.module.visual_model.patdis = 'patient_series_discrimination' in config['train']
        if model.module.visual_model.patdis is not None and model.module.visual_model.patdis:
            if not isinstance(model.module.patdistemperature,torch.nn.Parameter):
                model.module.patdistemperature[0] = config['train']['patdis_init_temperature']
                model.module.patdistemperature = torch.nn.Parameter(model.module.patdistemperature).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr = config['train']['learning_rate'], weight_decay = config['train']['weight_decay'])


    cumbatchnum = 1
    if 'cumbatchnum' in config['train'] and config['train']['cumbatchnum'] > 1:
        cumbatchnum = config['train']['cumbatchnum']
    saveepochinterval = 1
    if 'saveepochinterval' in config['train']:
        saveepochinterval = config['train']['saveepochinterval']
    if ('early_stop_step' in config['train']) and (len(train_dataset) > config['train']['early_stop_step']*config['data']['batch_size']*cumbatchnum):
        train_sub_dataset = SubDataset(train_dataset,config['train']['early_stop_step']*config['data']['batch_size']*cumbatchnum) # a subdataset if we do early stop
    else:
        train_sub_dataset = train_dataset
    train_loader = torch.utils.data.DataLoader(train_sub_dataset,batch_size=config['data']['batch_size'],collate_fn = collate(config['data']['text_max_tokens'],patching,'cpu',0),shuffle=True,num_workers=config['train']['num_train_loader_workers'],timeout=7200)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=config['data']['batch_size'],collate_fn = collate(config['data']['text_max_tokens'],patching,'cpu',0),shuffle=False,num_workers=config['train']['num_val_loader_workers'],timeout=1200)

    for e in range(config['train']['epochs']):

        totalloss = 0.0
        totals = 0
        totalpatdisloss = 0.0
        totalpatdisagg = 0.0
        gradnans = 0
        taccs1,iaccs1,taccs5,iaccs5 = [],[],[],[]

        # if using subdataset, we need to resample before beginning of training
        if 'early_stop_step' in config['train'] and isinstance(train_sub_dataset,SubDataset):
            train_sub_dataset.resample()

        try:
            cumulator = []
            for batchnum,batch in enumerate(tqdm(train_loader)):
                cumulator.append(batch) 
                if  cumbatchnum > 1:
                    if len(cumulator) < cumbatchnum: # cumulate a few batches for processing together
                        continue

                model.train()
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                    out1list, out2list,patdislist,mlist,hnlist = [],[],[],[],[]
                    for z,batchz in enumerate(cumulator):
                        outs = model(batchz)
                        out1 = outs[0]
                        out2 = outs[1]
                        out1list.append(out1)
                        if patdis is not None:
                            out2list.append(out2[0])
                            patdislist.append(out2[1])
                            mlist.append(out2[2])
                        else:
                            out2list.append(out2)
                    out1 = torch.cat(out1list,dim=0)
                    out2 = torch.cat(out2list,dim=0)
                    loss,losstemp = clip_objective(out1,out2,model.module.temperature)
                    if patdis is not None:
                        patdisres = torch.cat(patdislist,dim=0)
                        mlist = torch.cat(mlist,dim=0)
                        assert len(mlist) == len(patdisres)
                        patdisloss,patdisagg = patdis_objective(patdisres,mlist,model.module.patdistemperature)

                if patdis is not None:
                    (losstemp+patdisloss*patdis).backward()
                else:
                    losstemp.backward()
                cumulator = []
                # gradient clipping
                if 'gradient_clip' in config['train']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['gradient_clip'])
                
                # check nan
                gradnan = False
                if 'checkgradnan' in config['train']:
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                param.grad = torch.nan_to_num(param.grad,nan=0.0,posinf = 0.0,neginf = 0.0)
                                gradnan = True
                                #break
                if gradnan: 
                    gradnans += 1
                    print('warning: grad nan')
                
                # deal with warmup 
                if e == 0 and 'warmup' in config['train'] and batchnum < config['train']['warmup'] * cumbatchnum:
                    warmup = (batchnum+1) / config['train']['warmup']
                    for g in optimizer.param_groups:
                        g['lr'] = config['train']['learning_rate'] * warmup
                
                optimizer.step()

                # calculate training metrics
                if len(out1) == config['data']['batch_size'] * cumbatchnum:
                    totalloss += loss.item()
                    tolog = {'loss':loss.item(),'temperature loss':losstemp.item(),'temperature':model.module.temperature.item(),'patdisloss': 0 if patdis is None else patdisloss.item(),'patdistemperature': 0 if patdis is None else model.module.patdistemperature.item(),'patdisagg': 0 if patdis is None else patdisagg.item()}
                    
                    
                    totals += 1
                    if patdis is not None:
                        totalpatdisloss += patdisloss.item()
                        totalpatdisagg += patdisagg.item()
                    with torch.no_grad():
                        tacc,iacc,_,_ = retrievaleval(out1,out2,1)
                        taccs1.append(tacc)
                        iaccs1.append(iacc)
                        tacc,iacc,_,_ = retrievaleval(out1,out2,5)
                        taccs5.append(tacc)
                        iaccs5.append(iacc)


        except RuntimeError as e: # if errored out, figure out why. If just dataloader killed unexpectedly, we can ignore
            print(e)
            time.sleep(100)
            traceback.print_exc()
            if 'early_stop_step' in config['train']:
                train_sub_dataset = SubDataset(train_dataset,config['train']['early_stop_step']*config['data']['batch_size']*cumbatchnum) # a subdataset if we do early stop
            else:
                train_sub_dataset = train_dataset
            train_loader = torch.utils.data.DataLoader(train_sub_dataset,batch_size=config['data']['batch_size'],collate_fn = collate(config['data']['text_max_tokens'],patching,'cpu',0),shuffle=True,num_workers=config['train']['num_train_loader_workers'],timeout=3600)
            continue
        taccs1 = torch.FloatTensor(taccs1)
        iaccs1 = torch.FloatTensor(iaccs1)
        taccs5 = torch.FloatTensor(taccs5)
        iaccs5 = torch.FloatTensor(iaccs5)
        print('train set text retrieval top 1 accuracy: '+str(torch.mean(taccs1).item()))
        print('train set image retrieval top 1 accuracy: '+str(torch.mean(iaccs1).item()))
        print('train set text retrieval top 5 accuracy: '+str(torch.mean(taccs5).item()))
        print('train set image retrieval top 5 accuracy: '+str(torch.mean(iaccs5).item()))
        print('epoch '+str(e)+' train loss: '+str(totalloss/totals))

        # validation
        with torch.no_grad():
            model.eval()
            embs1 = []
            embs2 = []
            mress = []
            patdisscores = []
            try:
                for batch in tqdm(val_loader):

                    with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                        outs = model(batch)
                    out1 = outs[0]
                    out2 = outs[1]
                    embs1.append(out1)
                    if patdis is not None:
                        embs2.append(out2[0])
                        patdisscores.append(out2[1])
                        mress.append(out2[2])
                    else:
                        embs2.append(out2)
            except RuntimeError as ex:
                print(ex)
                traceback.print_exc()
                val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=config['data']['batch_size'],collate_fn = collate(config['data']['text_max_tokens'],patching,'cpu',0),shuffle=False,num_workers=config['train']['num_val_loader_workers'],timeout=1200)
                continue
            # compute validation metrics and statistics over the entire val set
            embs1 = torch.cat(embs1,dim=0)
            embs2 = torch.cat(embs2,dim=0)
            loss,_ = clip_objective(embs1,embs2)
            print("Val scaled loss: "+str(loss.item()))
            tacc,iacc, _, _ = retrievaleval(embs1,embs2,1)
            print('val set text retrieval top 1 accuracy: '+str(tacc))
            print('val set image retrieval top 1 accuracy: '+str(iacc))
            print("Val scaled loss: "+str(loss.item()))
            tacc5,iacc5, _, _ = retrievaleval(embs1,embs2,5)
            print('val set text retrieval top 5 accuracy: '+str(tacc5))
            print('val set image retrieval top 5 accuracy: '+str(iacc5))
            valclipscore = clipscore(embs1,embs2)
            print('val set clip score: ' +str(valclipscore))
             

            
            if patdis is not None:
                mress = torch.cat(mress,dim=0)
                patdisscores = torch.cat(patdisscores,dim=0)
                patdisloss,patdisagg = patdis_objective(patdisscores,mress,model.module.patdistemperature)
                print('val set patdis loss: '+str(patdisloss.item()))
                print('val set patdis agg: '+str(patdisagg.item()))

            torch.save(model,config['train']['ckptsavedir']+'/last.pt')
            if e % saveepochinterval == 0:
                torch.save(model,config['train']['ckptsavedir']+'/'+str(e)+'.pt')

if __name__ == '__main__':
    cf_fd = parse_args()
    config = yaml.load(cf_fd, Loader = yaml.FullLoader)
    main(config)


