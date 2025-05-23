# Prima training and evaluation scripts

![WeChat Image_20250205152113](https://github.com/user-attachments/assets/9da9490f-8ac9-4fca-89e3-7ea3fcf097e6)
![WeChat Image_20250205152226](https://github.com/user-attachments/assets/a5330ae0-ad8c-4d03-aac2-f9559fdc22fa)
In this directory, we include scripts for replicating our main CLIP training process, as well as classification head training, validation, and prospective test-set evaluation. In addition, we include scripts for performing LIME visualizations.

Please note that the scripts in this directory assumes that the raw data has already been processed and tokenized through VQ-VAE tokenizer (see `preprocessing and tokenization` directory for details and scripts for doing this step). 

The scripts in this directory should be run with this directory as working directory.

Note: "MRI Sequence" and "MRI series" are interchangeable terms in describing one entire 3D volume of MRI scan. We consistently used "sequence" in the paper, but the terms are mixed in the code repository (especially in variable namings). 

## Data preparation

Due to privacy policies, we are not allowed to share any raw patient imaging data in this repository. Therefore, we demonstrate all of our codes using generated fake data. To generate the fake data, simply run 
```
python generate_fake_data.py
```
The script will automatically generate 2000 fake studies as retrospective training set and another 2000 fake studies as prospective test set. It also generates fake labels and fake summarized reports.

We go over the detailed data structure below, in case you want to replicate our scripts with your own real data. If you simply wish to run our scripts on the generated fake data, you can skip to the remaining of this section and go directly to the [training section](https://github.com/MLNeurosurg/Prima/tree/main/Prima%20training%20and%20evaluation#clip-training).

### Tokenized data format

The tokenized MRI sequences should be stored in the following file structure:
```
<name of study>
├── <name of sequence 1>
│   └── emb
│       └── <name of VQ-VAE tokenizer>
│           ├── emb_meta.json
│           └── stacked
│               └── stacked.pt
├── <name of sequence 2>
│ ...
```
For example, in the fake generated data, the folder `fake_data/data/BRAIN_FAKE_20999` has the following structure:
```
BRAIN_FAKE_20999
├── AX_3D_T1
│   └── emb
│       └── FAKE_TOKENIZER
│           ├── emb_meta.json
│           └── stacked
│               └── stacked.pt
├── AX_DISC_3
│   └── emb
│       └── FAKE_TOKENIZER
│           ├── emb_meta.json
│           └── stacked
│               └── stacked.pt
├── AX_T1W_IR
│   └── emb
│       └── FAKE_TOKENIZER
│           ├── emb_meta.json
│           └── stacked
│               └── stacked.pt
├── AX_T1W_R
│   └── emb
│       └── FAKE_TOKENIZER
│           ├── emb_meta.json
│           └── stacked
│               └── stacked.pt
└── lt_SAG
    └── emb
        └── FAKE_TOKENIZER
            ├── emb_meta.json
            └── stacked
                └── stacked.pt
```
The `stacked.pt` under each sequence folder needs to be a file resulting from a `torch.save` that stores the VQ-VAE encoded tokens for this sequence stacked into a tensor. The size of the tensor should be one of the following, depending on the original orientation of the sequence
```
<num tokens>x2x2x8x8
<num tokens>x2x8x2x8
<num tokens>x2x8x8x2
```
`emb_meta.json` contains information about the embeddings in the `stacked.pt` file. The structure of `emb_meta.json` needs to be this:
```
{
  "PaddedVolShape": [x,y,z] (where x, y, z are dimensions of each individual token in before VQ-VAE, e.g. 32x32x4)
  "PatchShape": [x,y,z] (where x,y,z are dimensions of the entire 3D Volume before VQ-VAE, e.g. 256x256x76)
  "OtsuThresholds":{
    "0": { (Tokens determined as 0% foreground)
        "OutfillCoords":[[a1,[x1,y1,z1]],[a2,[x2,y2,z2]],...] (Each a indicate the index in stack.pt where this token is stored; each x,y,z indicate the coordinates of this token; the entire "OutfillCoords" indicates a list of tokens that are determined as 0% foreground)
        "InfillCoords":[[a1,[x1,y1,z1]],[a2,[x2,y2,z2]],...] (additional tokens isolated from remaining parts of MRI after removing the OutfillCoords tokens. Should also be removed from the masks)
         }
    "1": ...
    ...
  }
  "emb_index": {"0":[x0,y0,z0],"1":[x1,y1,z1],...} (maps each index in stacked.pt to 3D coordinates)
}
```
The entire "OtsuThresholds" is used to remove tokens that are mostly empty from the input to reduce memory usage. If you do not with to remove the empty tokens, you can put all tokens after "100" threshold and empty lists for all lower thresholds.

### Data Json

The tokenized data information above needs to be summarized in a json file, e.g. `fake_data/datajson.json`. The format of this json file needs to be the following:
```
[
    [
        "<path to study data folder>",
        [
            ["<name of sequence 1>", [0,0,0,0,0,0]], (the 6 numbers at the end are deprecated and not used, but needs to be there to preserve the structure)
            ["<name of sequence 2>", [0,0,0,0,0,0]],
            ...
        ],
        "<full unsummarized radiology report of study>",
        "<study description>"
    ],
    ...
]
```
The training program will load in this data json file to determine what studies to include in the training. We generate a data json for training set and another one for prospective test set.

### Data Labels

For each of the 52 classification tasks, we need to have labels for each study. We provide the list of positive studies for each label in a `txt` file, where each row is the name of one positive study. We provide the label txts of the fake training set in `fake_data/retrospective_classification` and the prospective set in `fake_data/prospective_classification`.

### Summarized reports

The summarized reports should be stored in a csv with two columns, the first being the name of the study and the second being the shortened report. See `fake_data/shortenedreports.csv` for example.

## CLIP Training

### (Optional) Sequence name encoder CLIP pretraining

To perform CLIP pre-training for the sequence name encoder, use the following command:

```
python seriename_clip.py --config configs/config-seriename-clip.yaml
```

You can modify training configurations in `configs/config-seriename-clip.yaml`. The resulting checkpoints should be in `serieclip_ckpts` folder unless you changed the ckpt save location in the config.

### Main CLIP training

If you have skipped the optional sequence name encoder CLIP pretraining, please comment out line 37 in `configs/config-main-clip-train.yaml` to not include a pre-trained sequence name encoder.

To perform main CLIP training, use the following command:

```
python clip_main.py --config configs/config-main-clip-train.yaml
```

You can modify training configuration in `configs/config-main-clip-train.yaml`. Follow the comments in the config file for guidance. The resulting checkpoints should be in `ckpts/` folder unless you changed the ckpt save location in the config.

### Downstream Classification Evaluation

To train and validate classification heads for downstream classification tasks, run the following command:

```
python classification_altogether.py --config configs/config-classification-head-train.yaml
```

By default it will use `ckpts/last.pt` as the visual encoder. The script will save a set of best checkpoints for each individual task for prospective evaluation.

To perform evaluation on the prospective test set, run the following command:

```
python eval_prospective_classification.py --config configs/config-prospective-classification.yaml
```

You can specify which head checkpoints you want to use in `configs/jsons/prospective_eval.json`.

## LIME visualization

To run LIME visualization on a specific datapoint for a checkpoint on a certain task, you can use `lime_visualization.ipynb`. It will generate a json file that contains the LIME importance score of each volume token, in the same order as the tokens are listed in `stacked.pt`.
