import torch
import torch.nn as nn
import numpy as np
import os
from numpy.typing import NDArray
from typing import Tuple, List, Dict
from tqdm import tqdm

from aria.embedding import get_global_embedding_from_midi


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


def get_aria_embeddings(model: nn.Module,
                        device: str,
                        evaluation_dir: str | None = None,
                        evaluation_path_dict: Dict[str, List] | None = None) -> Dict[str, Dict]:
    """Constructs Aria embeddings for performances.

    Args:
        model: Aria model.
        device: Device.
        evaluation_dir: A directory with score subdirectories and performances
          within them.
        evaluation_path_dict: A dictionary with score names as keys and lists
          of performance paths as values. A more flexible alternative to
          `evaluation_dir`.

    Return:
        A dictionary with score names as keys.
    """
    emb_dict = {}

    if evaluation_dir is not None:
        evaluation_path_dict = {}

        for score_name in os.listdir(evaluation_dir):
            score_dir = os.path.join(evaluation_dir, score_name)
            if not os.path.isdir(score_dir):
                continue

            evaluation_path_dict[score_name] = []
            for perf_name in os.listdir(score_dir):
                if perf_name.endswith(".mid") or perf_name.endswith(".midi"):
                    perf_path = os.path.join(score_dir, perf_name)  
                    evaluation_path_dict[score_name].append(perf_path)

    for score_name in tqdm(evaluation_path_dict):
        emb_dict[score_name] = {}

        for perf_path in evaluation_path_dict[score_name]:
            emb = get_global_embedding_from_midi(
                model=model,
                midi_path=perf_path,
                device=device,
            )
            emb_dict[score_name][perf_path] = emb.unsqueeze(0).cpu()
    
    return emb_dict
