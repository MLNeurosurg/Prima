import csv,json,sys,os,random
import torch
from tqdm import tqdm

random.seed(9174)

# This script will generate a total of 1000 fake studies totaling about ~10G in size


# make data directory
try:
    os.mkdir('fake_data')
    os.mkdir('fake_data/data')
except:
    pass

from util_fake_data import sequence_names,fake_report,study_desc,shortened_reports, classes
study_desc = list(study_desc)
# make a single fake study
def fake_study_gen(hash):
    os.mkdir('fake_data/data/'+hash)
    seqnum = random.randint(4,8) # randomly determine the number of sequences 
    seqnames = random.sample(sequence_names,k=seqnum) # determine name of sequences
    seqs = []
    ret = ['fake_data/data/'+hash,seqs,hash+' '+fake_report,random.choice(study_desc)]
    for name in seqnames:
        try:
            os.mkdir('fake_data/data/'+hash+'/'+name)
            os.mkdir('fake_data/data/'+hash+'/'+name+'/emb')
            os.mkdir('fake_data/data/'+hash+'/'+name+'/emb/FAKE_TOKENIZER')
        except:
            continue
        seqlen = random.randint(8,20)*4 # number of images in sequence
        orientation = 0 # 0 for axial, 1 for saggital, 2 for coronal
        pdshape = [4,32,32]
        ptshape = [seqlen,256,256]
        if 'sag' in name.lower():
            orientation = 1
            pdshape = [32,4,32]
            ptshape = [256,seqlen,256]
        elif 'cor' in name.lower():
            orientation = 2
            pdshape = [32,32,4]
            ptshape = [256,256,seqlen]

        # create emb_meta
        coords = [] # get coordinates for each volume token
        for i in range(0,seqlen,4):
            for j in range(0,256,32):
                for k in range(0,256,32):
                    if orientation == 0:
                        coords.append([i,j,k])
                    elif orientation == 1:
                        coords.append([j,i,k])
                    else: 
                        coords.append([j,k,i])
        emb_index = {}
        for i,coord in enumerate(coords):
            emb_index[str(i)] = coord
        enumcoords = [[i,o] for i,o in enumerate(coords)]
        sublists = divide_into_sublists(enumcoords)
        otsuthresh = {}
        for i, sublist in enumerate(sublists):
            d = {'OutfillCoords':sublist, 'InfillCoords':[]}
            otsuthresh[str(i)] = d
        emb_meta = {'PaddedVolShape':pdshape,'PatchShape':ptshape,'OtsuThresholds':otsuthresh,'emb_index':emb_index}
        json.dump(emb_meta,open('fake_data/data/'+hash+'/'+name+'/emb/FAKE_TOKENIZER/emb_meta.json','w+'),indent=2)

        # generate stacked.pt
        shape = [2,8,8,8]
        shape[orientation+1]=2
        stacked = torch.randn(len(emb_index),shape[0],shape[1],shape[2],shape[3])
        os.mkdir('fake_data/data/'+hash+'/'+name+'/emb/FAKE_TOKENIZER/stacked')
        torch.save(stacked,'fake_data/data/'+hash+'/'+name+'/emb/FAKE_TOKENIZER/stacked/stacked.pt')

        seqs.append([name,[0,0,0,0,0,0]])

    return ret

def divide_into_sublists(data, num_sublists=101):
    # Shuffle the list to ensure randomness
    random.shuffle(data)
    
    # Create a list of empty lists for sublists
    sublists = [[] for _ in range(num_sublists)]
    
    # Distribute elements to sublists
    for index, element in enumerate(data):
        sublist_index = index % num_sublists
        sublists[sublist_index].append(element)
    
    return sublists

# generate data json
datajson = []
hashes = []
for i in tqdm(range(2000)):
    h = 'BRAIN_FAKE_'+str(10000+i)
    ret = fake_study_gen(h)
    hashes.append(h)
    datajson.append(ret)
json.dump(datajson,open('fake_data/datajson.json','w+'),indent=2)

# generate shortened report csv
writer = csv.writer(open('fake_data/shortenedreports.csv','w+'))
for data in datajson:
    hash = data[0].split('/')[-1]
    writer.writerow([hash,random.choice(shortened_reports)])

# generate fake classification data and val splits
os.mkdir('fake_data/retrospective_classification')
for classname in classes:
    num = random.randint(400,800)
    poses = random.sample(hashes[100:1950],k=num)
    f=open('fake_data/retrospective_classification/'+classname+'.txt','w+')
    f.write('\n'.join(poses))
    f.close()
    negs = []
    i = 0
    while len(negs) < 100:
        if hashes[i] not in poses:
            negs.append(hashes[i])
        i += 1
    vals = poses[0:100] + negs
    f=open('fake_data/retrospective_classification/'+classname+'_val.txt','w+')
    f.write('\n'.join(vals))
    f.close()

# generate fake prospective data json
datajson = []
hashes = []
for i in tqdm(range(2000)):
    h = 'BRAIN_FAKE_'+str(20000+i)
    ret = fake_study_gen(h)
    hashes.append(h)
    datajson.append(ret)
json.dump(datajson,open('fake_data/datajson-prospective.json','w+'),indent=2)

# generate shortened report csv for prospective test set
writer = csv.writer(open('fake_data/shortenedreportsprospective.csv','w+'))
for h in hashes:
    writer.writerow([h,random.choice(shortened_reports)])

# generate fake classification data for prospectives
os.mkdir('fake_data/prospective_classification')
for classname in classes:
    num = random.randint(400,800)
    poses = random.sample(hashes[100:1950],k=num)
    f=open('fake_data/prospective_classification/'+classname+'.txt','w+')
    f.write('\n'.join(poses))
    f.close()

# generate config json for prospective evals
d = {}
for i,classname in enumerate(classes):
    d[classname] = ['fake_data/prospective_classification/'+classname+'.txt',[['2025-fake-data-heads/bestauc_'+classname+'.pt',i]]]
json.dump(d,open('fake_data/prospective-config.json','w+'),indent=2)
