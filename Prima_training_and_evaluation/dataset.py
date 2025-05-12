import torch
import random
import time
import json
import csv
import os
import sys
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import nibabel as nib
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer


# Character to index mapping for series names
CHAR_TO_INDEX = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz_+-0123456789.*(),")}


def chartovec(s: str) -> torch.Tensor:
    """Convert a string to a tensor of character indices.
    
    Args:
        s: Input string to convert
        
    Returns:
        Tensor of character indices, with unknown characters mapped to index 45
        and an end token (46) appended
    """
    ret = []
    for c in s.lower():
        try:
            ret.append(CHAR_TO_INDEX[c] + 1)
        except KeyError:
            ret.append(45)  # Unknown character index
    ret.append(46)  # End token
    return torch.LongTensor(ret)


def preprocess_text(text: str, split_finding: bool = False) -> str:
    """Preprocess full reports by removing unnecessary text.
    
    Args:
        text: Input text to preprocess
        split_finding: Whether to split on findings section
        
    Returns:
        Preprocessed text
    """
    if split_finding:
        for section in ['FINDINGS:', 'Findings:', 'INTERPRETATION:']:
            if section in text:
                text = text[text.index(section):]
                break
                
    # Remove dictation information
    if 'Dictated by:' in text:
        text = text[:text.rindex('Dictated by:')]
    return text


def preprocess_shortened_text(text: str, text_limit: int, tokenizer: Any, 
                            is_train: bool) -> str:
    """Preprocess shortened reports by organizing into list items.
    
    Args:
        text: Input text to preprocess
        text_limit: Maximum token length
        tokenizer: Text tokenizer
        is_train: Whether in training mode
        
    Returns:
        Preprocessed shortened text
    """
    # Split into list items
    items = []
    for line in text.split('\n'):
        if len(line) < 3:
            continue
        if '. ' not in line:
            items.append(line)
        else:
            items.append(line[line.index('. ') + 2:])
            
    # Shuffle items if training
    if is_train:
        random.shuffle(items)
        
    # Remove items if text is too long
    while True:
        ret = ''
        for i, item in enumerate(items):
            ret += f"{i+1}. {item}\n"
        if len(tokenizer(ret)) < text_limit:
            break
        items = items[:-1]
    return ret[:-1]


def convert_serienames_to_tensor(serienames: List[List[str]]) -> torch.Tensor:
    """Convert series names to tensor format.
    
    Args:
        serienames: List of series names
        
    Returns:
        Tensor of encoded series names
    """
    # Find max dimensions
    max1 = max(len(b1) for b1 in serienames)
    max2 = max(len(b2) for b1 in serienames for b2 in b1)
    
    # Create tensor and fill
    ret = torch.zeros(len(serienames), max1, max2, dtype=torch.long)
    for i, batch in enumerate(serienames):
        for j, name in enumerate(batch):
            ret[i, j, :len(name)] = torch.tensor([ord(c) for c in name])
    return ret


def filter_coords(meta: Dict, percentage_to_use: int, embs: torch.Tensor, 
                 fill_hole: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter coordinates based on pixel intensity.
    
    Args:
        meta: Metadata containing coordinate information
        percentage_to_use: Percentage threshold for filtering
        embs: Embeddings tensor
        fill_hole: Whether to fill holes in the data
        
    Returns:
        Tuple of (filtered embeddings, filtered positions, position map)
    """
    percentage_meta = meta['OtsuThresholds']
    uses = []
    
    if fill_hole:
        meta_dict = {
            x*1000000 + y*1000 + z: int(idx)
            for idx, (x, y, z) in meta['emb_index'].items()
        }
        
    # Collect coordinates above threshold
    for i in range(percentage_to_use, 101):
        uses.extend(percentage_meta[str(i)]['OutfillCoords'])
        if fill_hole and i <= 20:
            infill_coords = percentage_meta[str(i)]['InfillCoords']
            uses.extend([(meta_dict[b[0]*1000000 + b[1]*1000 + b[2]], b) 
                        for b in infill_coords])
                        
    # Convert to tensors
    use_ids = torch.LongTensor([u[0] for u in uses])
    emb_pos = torch.LongTensor([u[1] for u in uses])
    
    return embs[use_ids], emb_pos, use_ids


def collate_fn(max_tokens: int, patchify: Callable, device: str, 
              text_pad_token_id: int, put_to_device: bool = False) -> Callable:
    """Create collate function for ProtoDataset.
    
    Args:
        max_tokens: Maximum text tokens allowed
        patchify: Patchification module
        device: Device to put tensors on
        text_pad_token_id: Token ID for padding
        put_to_device: Whether to move tensors to device
        
    Returns:
        Collate function
    """
    def collate(datas: List[Dict]) -> Dict[str, torch.Tensor]:
        # Initialize tracking variables
        text_max = 0
        study_img_max = 0
        serie_img_maxs = [0] * 1000
        
        # Process each data point
        datas_new = []
        study_lens = []
        serienames = []
        hashnames = []
        study_descriptions = []
        
        for d in datas:
            hashnames.append(d['hash'])
            serienames.append(d['serienames'])
            study_descriptions.append(d['studydescription'])
            
            coords = d.get('coordinates')
            data_new = [
                patchify(d['visual'], coords=coords),
                d['text'],
                d['textlen']
            ]
            
            visuals, text, textlen = data_new
            datas_new.append(data_new)
            
            # Update maximum lengths
            text_max = max(textlen, text_max)
            for i, im in enumerate(visuals):
                serie_img_maxs[i] = max(len(im), serie_img_maxs[i])
            study_lens.append(len(visuals))
            study_img_max = max(len(visuals), study_img_max)
            
        # Convert series names to tensor
        serienames = convert_serienames_to_tensor(serienames)
        study_lens = torch.LongTensor(study_lens)
        
        # Process visuals
        visuals = [[] for _ in range(study_img_max)]
        serie_lenss = [[] for _ in range(study_img_max)]
        
        # Process texts
        texts = []
        text_lens = []
        
        for visual, text, textlen in datas_new:
            # Process each series
            for i, im in enumerate(visual):
                sizes = list(im.shape)
                h = sizes[0]
                img_pad_len = serie_img_maxs[i] - h
                sizes[0] = img_pad_len
                img_pad = torch.zeros(sizes)
                visuals[i].append(torch.cat([im, img_pad], dim=0))
                serie_lenss[i].append(len(im))
                
            # Pad empty series
            for i in range(len(visual), study_img_max):
                sizes = list(visual[0].shape)
                sizes[0] = serie_img_maxs[i]
                visuals[i].append(torch.zeros(sizes))
                serie_lenss[i].append(0)
                
            # Process text
            text_pad_len = text_max - textlen
            text_pad = torch.full((text_pad_len,), text_pad_token_id)
            texts.append(torch.cat([text, text_pad], dim=0))
            text_lens.append(textlen)
            
        # Process study descriptions
        sd_max_len = max(len(tensor) for tensor in study_descriptions)
        study_desc = torch.zeros(len(study_descriptions), sd_max_len)
        for i, tensor in enumerate(study_descriptions):
            study_desc[i, :len(tensor)] = tensor
            
        # Move tensors to device if needed
        if device != 'cpu' and not put_to_device:
            device = 'cpu'
            
        # Create return dictionary
        ret_dict = {
            'text': torch.stack(texts, dim=0).long().to(device),
            'textlen': torch.LongTensor(text_lens).to(device),
            'serienames': serienames.to(device),
            'hash': hashnames,
            'studydescription': study_desc
        }
        
        serie_lenss = torch.LongTensor(serie_lenss).transpose(0, 1).to(device)
        ret_dict['visual'] = [torch.stack(ims, dim=0).to(device) for ims in visuals]
        ret_dict['lens'] = study_lens.to(device)
        ret_dict['lenss'] = serie_lenss
        
        return ret_dict
    return collate


def collate_visual_hash(patchify: Callable, device: str, 
                       use_labels: bool = False, 
                       put_to_device: bool = False) -> Callable:
    """Create collate function for visual and hash data.
    
    Args:
        patchify: Patchification module
        device: Device to put tensors on
        use_labels: Whether to include classification labels
        put_to_device: Whether to move tensors to device
        
    Returns:
        Collate function
    """
    def collate(datas: List[Dict]) -> Dict[str, torch.Tensor]:
        # Initialize tracking variables
        study_img_max = 0
        serie_img_maxs = [0] * 1000
        
        # Process each data point
        datas_new = []
        study_lens = []
        serienames = []
        labels = []
        study_descriptions = []
        
        for d in datas:
            serienames.append(d['serienames'])
            study_descriptions.append(d['studydescription'])
            
            coords = d.get('coordinates')
            patched = patchify(d['visual'], coords=coords)
            label = d['label'] if use_labels else None
            
            data_new = [patched, label, d['hash']]
            datas_new.append(data_new)
            
            # Update maximum lengths
            study_lens.append(len(patched))
            if len(patched) > study_img_max:
                study_img_max = len(patched)
            for i, im in enumerate(patched):
                serie_img_maxs[i] = max(len(im), serie_img_maxs[i])
                
        # Process visuals
        visuals = [[] for _ in range(study_img_max)]
        serie_lenss = [[] for _ in range(study_img_max)]
        
        # Process each data point
        strs = []
        labels = []
        for visual, label, s in datas_new:
            # Process each series
            for i, im in enumerate(visual):
                sizes = list(im.shape)
                h = sizes[0]
                img_pad_len = serie_img_maxs[i] - h
                sizes[0] = img_pad_len
                img_pad = torch.zeros(sizes)
                visuals[i].append(torch.cat([im, img_pad], dim=0))
                serie_lenss[i].append(len(im))
                
            # Pad empty series
            for i in range(len(visual), study_img_max):
                sizes = list(visual[0].shape)
                sizes[0] = serie_img_maxs[i]
                visuals[i].append(torch.zeros(sizes))
                serie_lenss[i].append(0)
                
            strs.append(s)
            labels.append(label)
            
        # Convert series names to tensor
        serienames = convert_serienames_to_tensor(serienames)
        
        # Process study descriptions
        sd_max_len = max(len(tensor) for tensor in study_descriptions)
        study_desc = torch.zeros(len(study_descriptions), sd_max_len)
        for i, tensor in enumerate(study_descriptions):
            study_desc[i, :len(tensor)] = tensor
            
        # Process labels if needed
        if use_labels:
            if isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels).long()
            else:
                labels = torch.LongTensor(labels)
                
        # Move tensors to device if needed
        if put_to_device:
            visuals = [[im.to(device) for im in ims] for ims in visuals]
            study_lens = torch.LongTensor(study_lens).to(device)
            serie_lenss = torch.LongTensor(serie_lenss).transpose(0, 1).to(device)
            serienames = serienames.to(device)
            study_desc = study_desc.to(device)
            
        return {
            'visual': [torch.stack(ims, dim=0) for ims in visuals],
            'lens': study_lens,
            'lenss': serie_lenss,
            'hash': strs,
            'serienames': serienames,
            'labels': labels,
            'studydescription': study_desc
        }
    return collate


class ProtoDataset(torch.utils.data.Dataset):
    """Base dataset class for medical imaging data."""
    
    def __init__(self, data_json: str, data_root_dir: str, text_max_len: int,
                 is_train: bool, tokenizer: str, vqvae_name: str,
                 pt_limit: int = 11, series_dropout_rate: float = 0.0,
                 split_finding_rate: float = 0.0, val_size: int = 254,
                 include_hash: bool = False, visual_hash_only: bool = False,
                 force_report_from_csv: Optional[str] = None,
                 percentage: Optional[int] = None, upsample_abnormal: int = 0,
                 no_visual_aug: bool = False, token_dropout: Optional[float] = None,
                 seriename_dropout: Optional[float] = None,
                 prospective_data_list: Optional[str] = None,
                 prospective: bool = False, exclude_series: List[str] = [],
                 no_split: bool = False, no_text_downstream: bool = False,
                 emb_name: str = 'emb', force_unk_seriename: bool = False,
                 serienum_check: bool = True):
        """Initialize the dataset.
        
        Args:
            data_json: Path to dataset JSON file
            data_root_dir: Root directory for embeddings
            text_max_len: Maximum text length in tokens
            is_train: Whether this is training data
            tokenizer: Text tokenizer to use
            vqvae_name: Name of VQVAE used for tokens
            pt_limit: Patient limit (unused)
            series_dropout_rate: Chance of dropping a series
            split_finding_rate: Chance of splitting findings
            val_size: Validation set size
            include_hash: Whether to include hash
            visual_hash_only: Whether to only return visual and hash
            force_report_from_csv: Path to CSV with reports
            percentage: Otsu cutoff percentage
            upsample_abnormal: Number of times to copy abnormal data
            no_visual_aug: Whether to disable visual augmentation
            token_dropout: Token dropout rate
            seriename_dropout: Series name dropout rate
            prospective_data_list: Data list for prospective dataset
            prospective: Whether this is prospective test set
            exclude_series: Series names to exclude
            no_split: Whether to skip validation split
            no_text_downstream: Whether this is a no-text task
            emb_name: Name of embedding folder
            force_unk_seriename: Whether to force unknown series names
            serienum_check: Whether to enforce minimum series count
        """
        # Load report CSV if provided
        self.report_csv_dict = None
        if force_report_from_csv:
            self.report_csv_dict = {}
            with open(force_report_from_csv) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        self.report_csv_dict[row[0]] = row[1]
                    else:
                        print(f'Warning: unreadable row in reportcsv: {row}')
                        
        # Load data JSON
        with open(data_json) as f:
            self.datas = json.load(f)
            
        # Split train/val if needed
        if not prospective and not no_split:
            if is_train:
                self.datas = self.datas[:-val_size]
            else:
                self.datas = self.datas[-val_size:]
                
        # Upsample abnormal data if needed
        if upsample_abnormal > 0 and is_train:
            from abnormaltextfilter import getabnormallist
            abnormal_list = set(getabnormallist(force_report_from_csv))
            new_datas = []
            for data in self.datas:
                study_path, _, _, _ = data
                hash_name = study_path.split('/')[-1]
                new_datas.append(data)
                if hash_name in abnormal_list:
                    for _ in range(upsample_abnormal):
                        new_datas.append(data)
            self.datas = new_datas
            
        # Set instance variables
        self.percentage = percentage
        self.is_train = is_train
        self.datalen = len(self.datas)
        self.text_max_len = text_max_len
        self.pt_limit = pt_limit
        self.no_visual_aug = no_visual_aug
        self.token_dropout = token_dropout
        self.seriename_dropout = seriename_dropout
        self.exclude_series = exclude_series
        self.emb_name = emb_name
        self.force_unk_seriename = force_unk_seriename
        
        # Initialize tokenizer
        if tokenizer == 'biomed':
            self.tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        elif tokenizer == 'tinyllama':
            self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        elif tokenizer == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            raise NotImplementedError(f"Tokenizer {tokenizer} not implemented")
            
        # Set additional variables
        self.vqvae_name = vqvae_name
        self.eos_id = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        self.serie_dropout = series_dropout_rate
        self.split_finding_rate = split_finding_rate
        self.include_hash = include_hash
        self.visual_hash_only = visual_hash_only
        
    def __len__(self) -> int:
        return self.datalen
        
    def get_hash(self, idx: int) -> str:
        return self.datas[idx][0].split('/')[-1]
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.getitem(idx)
        
    def getitem(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError
        
    def get_text(self, idx: int) -> str:
        if self.report_csv_dict is None:
            _, _, report, _ = self.datas[idx]
            return preprocess_text(report, split_finding=True)
        else:
            study_path, _, report, _ = self.datas[idx]
            hash_name = study_path.split('/')[-1]
            if hash_name not in self.report_csv_dict:
                print(f'Warning: {hash_name} has default report but no report from csv')
                report = preprocess_text(report, split_finding=True)
            else:
                report = self.report_csv_dict[hash_name]
            return preprocess_shortened_text(report, self.text_max_len,
                                          self.tokenizer, self.is_train)
                                          
    def get_text_dict(self, report: str, split_finding: bool) -> Dict[str, torch.Tensor]:
        if self.report_csv_dict is None:
            report = preprocess_text(report, split_finding=split_finding)
        else:
            report = preprocess_shortened_text(report, self.text_max_len,
                                            self.tokenizer, self.is_train)
                                            
        text = self.tokenizer(report + '<|endoftext|>')['input_ids']
        if len(text) > self.text_max_len:
            text = text[:self.text_max_len]
            text[-1] = self.eos_id
            textlen = self.text_max_len
        else:
            textlen = len(text)
            
        return {
            'text': torch.LongTensor(text),
            'textlen': textlen
        }
        
    def get_path(self, idx: int) -> str:
        study_path, _, _, _ = self.datas[idx]
        return study_path
        
    def find_by_hash(self, hash_name: str, get_id_only: bool = False) -> Union[Dict[str, Any], int]:
        for i, d in enumerate(self.datas):
            if hash_name == d[0].split('/')[-1]:
                if get_id_only:
                    return i
                return self.getitem(i)


class RachelDataset(ProtoDataset):
    """Dataset for new tokens from Rachel and Token256."""
    
    def getitem(self, idx: int, custom_input: Optional[Tuple] = None) -> Dict[str, Any]:
        if custom_input is not None:
            study_path, series, report, study_descr = custom_input
        else:
            study_path, series, report, study_descr = self.datas[idx]
            
        hash_name = study_path.split('/')[-1]
        
        # Handle split finding
        split_finding = random.random() < self.split_finding_rate
        
        # Handle report replacement
        if self.report_csv_dict is not None:
            if hash_name not in self.report_csv_dict:
                print(f'Warning: {hash_name} has default report but no report from csv')
                report = preprocess_text(report, split_finding=split_finding)
            else:
                report = self.report_csv_dict[hash_name]
                
        # Handle text-only mode
        if hasattr(self, 'textdictonly') and self.textdictonly:
            return self.get_text_dict(report, split_finding)
            
        # Process series
        out_series = []
        attempts = 0
        
        while len(out_series) == 0:
            if attempts >= 5:
                print(f'{hash_name} cannot get enough good series!')
                if self.is_train:
                    return self.getitem(random.randint(0, len(self) - 1))
                else:
                    return self.getitem(idx - 1)
            attempts += 1
            
            # Process each series
            out_series = []
            for serie, _ in series:
                if serie in self.exclude_series:
                    continue
                    
                # Handle series dropout
                if self.is_train and not self.no_visual_aug and self.serie_dropout > 0:
                    if random.random() < self.serie_dropout:
                        continue
                        
                # Load embeddings
                serie_path = f"{study_path}/{serie}/{self.emb_name}/{self.vqvae_name}"
                all_embs = torch.load(f"{serie_path}/stacked/stacked.pt",
                                    map_location='cpu')
                percentage_meta = json.load(open(f"{serie_path}/emb_meta.json"))
                
                # Adjust percentage if training
                percentage_adjust = 0
                if self.is_train and not self.no_visual_aug:
                    percentage_adjust = random.randint(-2, 2)
                    
                # Try different percentages
                for percent in range(self.percentage + percentage_adjust, -1, -1):
                    embs, emb_pos, pos_map = filter_coords(percentage_meta, percent,
                                                         all_embs)
                    if len(emb_pos) > 25 and percent > 0:
                        break
                    if percent == 0:
                        if len(emb_pos) > 0:
                            emb_pos = []
                        else:
                            embs = all_embs
                            emb_pos = []
                            if len(embs) > 5000 or len(embs) == 0:
                                break
                            emb_pos = torch.LongTensor([
                                percentage_meta['emb_index'][str(i)]
                                for i in range(len(embs))
                            ])
                            
                # Skip if no embeddings
                if len(emb_pos) == 0:
                    continue
                    
                # Check for NaN
                if torch.isnan(embs).any():
                    print('Warning: NaN values found in embeddings')
                    continue
                    
                # Handle token dropout
                if (self.is_train and self.token_dropout is not None and
                    not self.no_visual_aug):
                    mask = torch.bernoulli(
                        torch.ones(len(embs)) * (1 - self.token_dropout)
                    )
                    indexes = mask.nonzero().squeeze()
                    embs = embs[indexes]
                    emb_pos = emb_pos[indexes]
                    pos_map = pos_map[indexes]
                    
                out_series.append((embs, emb_pos, serie, None, pos_map))
                
        # Shuffle series if training
        if self.is_train and not self.no_visual_aug:
            random.shuffle(out_series)
            
        # Process series names
        serie_name_str = [o[2] for o in out_series]
        serie_names = []
        
        for o in out_series:
            if ((self.seriename_dropout is not None and
                 self.is_train and not self.no_visual_aug and
                 random.random() < self.seriename_dropout) or
                self.force_unk_seriename):
                serie_names.append(chartovec('unk'))
            else:
                serie_names.append(chartovec(o[2]))
                
        # Extract embeddings and positions
        out_series_pos = [o[1] for o in out_series]
        pos_maps = [o[4] for o in out_series]
        out_series = [o[0] for o in out_series]
        
        # Process study description
        study_descr = chartovec(study_descr)
        
        # Return visual-only data if requested
        if self.visual_hash_only:
            return {
                'visual': out_series,
                'hash': study_path.split('/')[-1],
                'serienames': serie_names,
                'coordinates': out_series_pos,
                'studydescription': study_descr,
                'posmap': pos_maps,
                'serienamestr': serie_name_str
            }
            
        # Process report
        if self.report_csv_dict is None:
            report = preprocess_text(report, split_finding=split_finding)
        else:
            report = preprocess_shortened_text(report, self.text_max_len,
                                            self.tokenizer, self.is_train)
                                            
        # Process text
        text = self.tokenizer(report + '<|endoftext|>')['input_ids']
        if len(text) > self.text_max_len:
            text = text[:self.text_max_len]
            text[-1] = self.eos_id
            textlen = self.text_max_len
        else:
            textlen = len(text)
            
        return {
            'visual': out_series,
            'text': torch.LongTensor(text),
            'textlen': textlen,
            'hash': hash_name,
            'serienames': serie_names,
            'coordinates': out_series_pos,
            'studydescription': study_descr,
            'posmap': pos_maps,
            'serienamestr': serie_name_str
        }


class SubDataset(torch.utils.data.Dataset):
    """Subset of a dataset with randomly selected samples."""
    
    def __init__(self, dataset: torch.utils.data.Dataset, limit: int):
        self.full_len = len(dataset)
        self.dataset = dataset
        self.ids = random.sample(range(self.full_len), limit)
        
    def __len__(self) -> int:
        return len(self.ids)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[self.ids[idx]]
        
    def resample(self) -> None:
        self.ids = random.sample(range(self.full_len), len(self.ids))
        
    def get_ratio(self) -> float:
        labels = sum(self.dataset.get_labels(self.ids[i])
                    for i in range(len(self)))
        neg_labels = len(self) - labels
        return neg_labels / labels


class SerieNameCLIPDataset(torch.utils.data.Dataset):
    """Dataset for training CLIP between series and series names."""
    
    def __init__(self, data_json: str, is_train: bool, vqvae_name: str,
                 token_dropout: float = 0, no_visual_aug: bool = False,
                 percentage: int = 5, val_size: int = 254,
                 special_book: Optional[List] = None, no_split: bool = False):
        # Load and split data
        with open(data_json) as f:
            orig_json = json.load(f)
            
        if no_split:
            my_orig_json = orig_json
        else:
            my_orig_json = orig_json[:-val_size] if is_train else orig_json[-val_size:]
            
        # Filter data if special book provided
        to_include_name_dict = {d[0]: d[1] for d in special_book} if special_book else None
        
        self.datas = []
        for data in my_orig_json:
            if to_include_name_dict is not None:
                h = data[0].split('/')[-1]
                if h not in to_include_name_dict:
                    continue
                self.datas.append((data[0], to_include_name_dict[h]))
            else:
                for serie in data[1]:
                    self.datas.append((data[0], serie))
                    
        # Set instance variables
        self.is_train = is_train
        self.token_dropout = token_dropout
        self.no_visual_aug = no_visual_aug
        self.vqvae_name = vqvae_name
        self.percentage = percentage
        
    def __len__(self) -> int:
        return len(self.datas)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        study_path, serie_o = self.datas[idx]
        
        # Handle series and orientation
        if isinstance(serie_o, str):
            serie = serie_o
            orientation = None
        else:
            serie, orientation = serie_o
            
        # Load embeddings
        serie_path = f"{study_path}/{serie}/emb/{self.vqvae_name}"
        all_embs = torch.load(f"{serie_path}/stacked/stacked.pt",
                            map_location='cpu')
        percentage_meta = json.load(open(f"{serie_path}/emb_meta.json"))
        
        # Adjust percentage if training
        percentage_adjust = 0
        if self.is_train and not self.no_visual_aug:
            percentage_adjust = random.randint(-2, 2)
            
        # Try different percentages
        for percent in range(self.percentage + percentage_adjust, -1, -1):
            embs, emb_pos, pos_map = filter_coords(percentage_meta, percent,
                                                 all_embs)
            if len(emb_pos) > 25 and percent > 0:
                break
            if percent == 0:
                if len(emb_pos) > 0:
                    emb_pos = []
                else:
                    embs = all_embs
                    emb_pos = []
                    if len(embs) > 5000 or len(embs) == 0:
                        break
                    emb_pos = torch.LongTensor([
                        percentage_meta['emb_index'][str(i)]
                        for i in range(len(embs))
                    ])
                    
        # Handle empty embeddings
        if len(emb_pos) == 0:
            return self[random.randint(0, len(self) - 1)]
            
        # Handle token dropout
        if (self.is_train and self.token_dropout is not None and
            not self.no_visual_aug):
            mask = torch.bernoulli(
                torch.ones(len(embs)) * (1 - self.token_dropout)
            )
            indexes = mask.nonzero().squeeze()
            embs = embs[indexes]
            emb_pos = emb_pos[indexes]
            pos_map = pos_map[indexes]
            
        hash_series = f"{study_path.split('/')[-1]}|{serie}"
        return embs, emb_pos, chartovec(serie), hash_series, orientation, pos_map




