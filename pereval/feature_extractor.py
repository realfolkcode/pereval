import torch
import torch.nn as nn
import numpy as np
import os
from numpy.typing import NDArray
from typing import Tuple, List, Dict
from tqdm import tqdm

from aria.embedding import get_global_embedding_from_midi, \
                           _validate_midi_for_emb, _get_chunks, \
                           MAX_EMBEDDING_SEQ_LEN, get_embedding_from_seq
from ariautils.tokenizer import AbsTokenizer
from ariautils.midi import MidiDict


class CLAMPExtractor:
    """Feature extractor for CLAMP3 model."""

    def extract_features(self,
                         emb_dir: str) -> Tuple[NDArray, List, List]:
        """Extracts features from a directory.

        Args:
            emb_dir: The directory where embeddings are stored.

        Returns:
            Embeddings of shape (N, D), a list of score names,
            and a list of performance names.
        """
        emb_lst = []
        score_lst = []
        perf_lst = []

        for score_name in os.listdir(emb_dir):
            score_dir = os.path.join(emb_dir, score_name)
            if not os.path.isdir(score_dir):
                continue

            for emb_name in os.listdir(score_dir):
                emb_path = os.path.join(score_dir, emb_name)
                if not emb_name.endswith('.npy'):
                    continue
                
                emb = np.load(emb_path)
                emb_lst.append(emb)

                perf_name = emb_name.split('.')[0]
                perf_name = f"{perf_name}.mid"

                score_lst.append(score_name)
                perf_lst.append(perf_name)
            
        embeddings = np.concatenate(emb_lst)
        
        return embeddings, score_lst, perf_lst


class FeatureExtractor:
    """Feature extractor for Aria model."""

    def extract_features(self,
                         emb_path: str) -> Tuple[NDArray, List, List]:
        """Extracts features from a .pt file.

        Args:
            emb_path: The path to embeddings file.

        Returns:
            Embeddings of shape (N, D), a list of score names,
            and a list of performance names.
        """
        embeddings = torch.load(emb_path)

        emb_lst = []
        score_lst = []
        perf_lst = []

        for score, perf_dict in embeddings.items():
            score_lst += [score] * len(perf_dict)
            perf_lst += list(perf_dict.keys())
            emb_lst += list(perf_dict.values())

        embeddings = np.concatenate(emb_lst)
        
        return embeddings, score_lst, perf_lst


def get_aria_embeddings(evaluation_dir: str,
                        model: nn.Module,
                        device,
                        is_first: bool = False) -> Dict:
    """Constructs Aria embeddings for performances.

    Args:
        evaluation_dir: A directory with performances.
        model: Aria model.
        device: Device.
        is_first: If True, computes embeddings only for the first chunk of a MIDI.

    Return:
        A dictionary with score names as keys.
    """
    emb_dict = {}

    for i, score_name in enumerate(tqdm(os.listdir(evaluation_dir))):
        score_dir = os.path.join(evaluation_dir, score_name)
        if not os.path.isdir(score_dir):
            continue

        emb_dict[score_name] = {}

        for perf_name in os.listdir(score_dir):
            if perf_name.endswith(".mid") or perf_name.endswith(".midi"):
                perf_path = os.path.join(score_dir, perf_name)
                if not is_first:
                    emb = get_global_embedding_from_midi(
                        model=model,
                        midi_path=perf_path,
                        device=device,
                    )
                else:
                    emb = get_first_embedding_from_midi(
                        model=model,
                        midi_path=perf_path,
                        device=device
                    )
                emb_dict[score_name][perf_name] = emb.unsqueeze(0).cpu()
    
    return emb_dict


def create_aria_seq_mask(seq: List) -> torch.tensor:
    """Creates a mask for Aria token sequence.

    (Experimental; not used in the paper)
    
    Args:
        seq: Sequence.
    
    Returns:
        A mask of shape (num_tokens,).
    """
    mask = torch.zeros(len(seq)).bool()
    for i in range(len(seq)):
        if seq[i][0] == 'piano':
            mask[i] = True
        elif seq[i][0] == 'onset':
            mask[i] = True
        elif seq[i][0] == 'dur':
            mask[i] = True
    return mask


@torch.inference_mode()
def get_full_embedding_from_midi(model,
                                 midi_path: str,
                                 notes_per_chunk: int = 300,
                                 device: str = "cuda",
                                 is_contrastive: bool = True) -> torch.tensor:
    """Calculates a full MIDI embedding with Aria.

    (Experimental; not used in the paper)

    Args:
        model: Aria model.
        midi_path: The path to a MIDI file.
        notes_per_chunk: The number of notes per chunk.
        device: Device.
        is_contrastive: If True, uses the contrastive embedding head.
    
    Returns:
        Embedding of shape (3 * num_notes, dim).
    """
    midi_dict = MidiDict.from_midi(mid_path=midi_path)

    tokenizer = AbsTokenizer()
    _validate_midi_for_emb(midi_dict)

    chunks = _get_chunks(midi_dict=midi_dict, notes_per_chunk=notes_per_chunk)
    seqs = [
        tokenizer.tokenize(c, add_dim_tok=False)[:MAX_EMBEDDING_SEQ_LEN]
        for c in chunks
    ]

    # Add back eos_tok if truncated by MAX_EMBEDDING_SEQ_LEN
    for seq in seqs:
        if seq[-1] != tokenizer.eos_tok:
            seq[-1] = tokenizer.eos_tok
    
    model.eval()

    emb_lst = []
    for seq in seqs:
        seq_enc = torch.tensor(tokenizer.encode(seq), device=device)
        seq_mask = create_aria_seq_mask(seq).to(device)
        if is_contrastive:
            emb = model.forward(seq_enc.view(1, -1))
        else:
            emb = model.model.forward(seq_enc.view(1, -1))
        emb = emb[:, seq_mask, :]
        assert emb.shape[1] <= 3 * notes_per_chunk
        
        emb_lst.append(emb)
    
    emb = torch.concat(emb_lst, 1).squeeze()

    return emb


@torch.inference_mode()
def get_first_embedding_from_midi(model,
                                  midi_path: str,
                                  notes_per_chunk: int = 300,
                                  device: str = "cuda") -> torch.tensor:
    """Calculates MIDI embedding for the first chunk with Aria.

    (Experimental; not used in the paper)

    Args:
        model: Aria model.
        midi_path: The path to a MIDI file.
        notes_per_chunk: The number of notes per chunk.
        device: Device.
    
    Returns:
        Embedding of shape (dim,).
    """
    midi_dict = MidiDict.from_midi(mid_path=midi_path)

    tokenizer = AbsTokenizer()
    _validate_midi_for_emb(midi_dict)

    chunks = _get_chunks(midi_dict=midi_dict, notes_per_chunk=notes_per_chunk)
    seqs = [
        tokenizer.tokenize(c, add_dim_tok=False)[:MAX_EMBEDDING_SEQ_LEN]
        for c in chunks
    ]
    seq = seqs[0]

    # Add back eos_tok if truncated by MAX_EMBEDDING_SEQ_LEN
    if seq[-1] != tokenizer.eos_tok:
        seq[-1] = tokenizer.eos_tok
    
    model.eval()

    emb = get_embedding_from_seq(model=model, seq=seq, device=device)
    emb = emb.squeeze()
    return emb
