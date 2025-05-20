import torch
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
from dataset import MrDataset, collatevisualhash


class FullMRIModel(torch.nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        # Load CLIP model
        self.clipmodel = torch.load(config['clip_ckpt'],
                                    map_location='cpu').module
        self.clipvisualmodel = self.clipmodel.visual_model
        self.clipvisualmodel.patdis = False

        # Initialize diagnosis and referral heads
        self.diagnosisheads = self._load_heads(config['diagnosis_heads_json'])
        self.referralheads = self._load_heads(config['referral_heads_json'])

        # Create ModuleLists for proper parameter registration
        self.diagnosis_modules = torch.nn.ModuleList(
            [head[0] for head in self.diagnosisheads.values()])
        self.referral_modules = torch.nn.ModuleList(
            [head[0] for head in self.referralheads.values()])

        # Load priority head
        self.priorityhead = torch.load(config['priority_head_ckpt'],
                                       map_location='cpu')

    def _load_heads(self,
                    json_path: str) -> Dict[str, Tuple[torch.nn.Module, int]]:
        """Load classification heads from JSON configuration."""
        heads = {}
        with open(json_path) as f:
            head_config = json.load(f)

        for name, (_, [(headpath, idx, thresh)]) in head_config.items():
            head = torch.load(headpath, map_location='cpu')
            head.thresh = float(thresh)
            heads[name] = (head, idx)

        return heads

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass through the model."""
        clip_embed = self.clipvisualmodel(x, retpool=True)

        retdict = {
            'diagnosis': {},
            'referral': {},
            'priority': {},
            'clip_emb': clip_embed.detach().cpu()
        }

        # Process diagnosis heads
        for name, (head, idx) in self.diagnosisheads.items():
            retdict['diagnosis'][name] = head(clip_embed)[:, idx] - head.thresh

        # Process referral heads
        for name, (head, idx) in self.referralheads.items():
            retdict['referral'][name] = head(clip_embed)[:, idx] - head.thresh

        # Process priority head
        priority_out = self.priorityhead(clip_embed)
        priority_levels = ['none', 'low', 'medium', 'high']
        retdict['priority'] = {
            level: priority_out[:, i]
            for i, level in enumerate(priority_levels)
        }

        return retdict

    @torch.no_grad()
    def forward_one_diag_only(self, x: torch.Tensor,
                              diagname: str) -> torch.Tensor:
        """Forward pass for a single diagnosis head."""
        clip_embed = self.clipvisualmodel(x, retpool=True)
        head, idx = self.diagnosisheads[diagname]
        return head(clip_embed)[:, idx] - head.thresh

    def make_no_flashattn(self) -> None:
        """Disable flash attention in the visual model."""
        self.clipvisualmodel.make_no_flashattn()


if __name__ == '__main__':
    # Example configuration
    config = {
        'clip_ckpt':
        'tempmodelsavesite/scratch/checkpoints96bigvit/55.pt',
        'diagnosis_heads_json':
        'configs/jsons/combbest927.json',
        'referral_heads_json':
        'configs/jsons/combbest107referral.json',
        'priority_head_ckpt':
        'tempmodelsavesite/104-priority-notord/bestauc2_priority.pt'
    }

    # Initialize model
    fullmodel = FullMRIModel(config)

    # Example inference
    import copy

    # Setup dataset and collator
    retrodataset = MrDataset(
        datajson='datajson/glmv8-3.json',
        datarootdir='/scratch/tocho_root/tocho1/yiweilyu/glmv8-3/',
        is_train=False,
        vqvaename=
        'TOKEN-MODEL-RESIZE-8-32-32-batch_permute-DATE-2024-04-02-0043AM',
        series_dropout_rate=0.0,
        percentage=5,
        novisualaug=True,
        nosplit=True,
        forcereportfromcsv='chatgpt/shortenedreportsglmv8full.csv',
        visualhashonly=True,
        text_max_len=128,
        tokenizer='biomed')

    collator = collatevisualhash(copy.deepcopy(
        fullmodel.clipmodel.patchifier).cpu(),
                                 'cuda:0',
                                 True,
                                 True,
                                 puttodevice=True)

    # Run inference
    with torch.no_grad(), torch.amp.autocast(device_type='cuda',
                                             dtype=torch.float16):
        fullmodel = fullmodel.half().cuda()
        sample = collator([retrodataset.find_by_hash('BRAIN_UM_3B6FE29D')])
        result = fullmodel(sample)
        print(result)
