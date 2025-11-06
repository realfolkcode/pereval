import torch
import torch.nn as nn
import numpy as np
import os
from numpy.typing import NDArray
from typing import Tuple, List, Dict
from tqdm import tqdm

from aria.embedding import get_global_embedding_from_midi


class FeatureExtractor:
    def extract_features(self,
                         emb_path: str) -> Tuple[NDArray, List]:
        """Extracts features from a .pt file.

        Args:
            emb_path: The path to embeddings file.

        Returns:
            Embeddings of shape (N, D) and a list of N labels.
        """
        embeddings = torch.load(emb_path)

        label_lst = []
        for label, emb in embeddings.items():
            label_lst += [label] * emb.shape[0]

        embeddings = np.concat([emb.numpy() for emb in embeddings.values()])
        return embeddings, label_lst


def get_aria_embeddings(evaluation_dir: str,
                        model: nn.Module,
                        device) -> Dict[str, torch.Tensor]:
    """Constructs Aria embeddings for performances.

    Args:
        evaluation_dir: A directory with performances.
        model: Aria model.
        device: Device.

    Return:
        A dictionary with score names as keys and performance embeddings
        as values.
    """
    emb_dict = {}

    for i, score_name in enumerate(tqdm(os.listdir(evaluation_dir))):
        score_dir = os.path.join(evaluation_dir, score_name)
        if os.path.isdir(score_dir):
            for perf_name in os.listdir(score_dir):
                if perf_name.endswith(".mid") or perf_name.endswith(".midi"):
                    perf_path = os.path.join(score_dir, perf_name)
                    emb = get_global_embedding_from_midi(
                        model=model,
                        midi_path=perf_path,
                        device=device,
                    )
                    if score_name not in emb_dict:
                        emb_dict[score_name] = emb.unsqueeze(0).cpu()
                    else:
                        emb_dict[score_name] = torch.cat((emb_dict[score_name],
                                                          emb.unsqueeze(0).cpu()), dim=0)
    
    return emb_dict
