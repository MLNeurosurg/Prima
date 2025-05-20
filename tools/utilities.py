import torch
import random
from typing import Any, List

# Character to index mapping for series names
CHAR_TO_INDEX = {
    char: idx
    for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz_+-0123456789.*(),")
}


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
