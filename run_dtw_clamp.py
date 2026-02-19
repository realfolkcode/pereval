import os
import argparse
import numpy as np
import torch
from dtaidistance import dtw, dtw_ndim, ed
from tqdm import tqdm
from typing import Dict, List


def load_embeddings(emb_dir):
    emb_lst = []

    for emb_name in os.listdir(emb_dir):
        if not emb_name.endswith(".npy"):
            continue
        
        emb_path = os.path.join(emb_dir, emb_name)
        emb = np.load(emb_path).squeeze()

        emb_lst.append(emb)

    return emb_lst


def calculate_dtw_distance(x,
                           y) -> float:
    """Calculates the distance between full embeddings.
    
    Args:
        x: Full performance embeddings.
        y: Full performance embeddings.
    
    Returns:
        The squared distance between full embeddings.
    """
    upper_bound = ed.distance(x, y, use_ndim=True)
    dist = dtw.distance_fast(x, y, max_dist=upper_bound, window=100, use_ndim=True)

    if dist == np.inf:
        dist = upper_bound

    dist = dist**2
    dist = dist / x.shape[1]
    dist = dist / ((x.shape[0] + y.shape[0]) / 2)

    return dist


def main(args):
    real_emb_dir = args.real_emb_dir
    gen_emb_dir = args.gen_emb_dir
    output_filename = args.output_filename

    output_path = os.path.join(gen_emb_dir, output_filename)

    dtw_dict = {}

    for score_name in tqdm(os.listdir(gen_emb_dir)):
        gen_score_dir = os.path.join(gen_emb_dir, score_name)
        if not os.path.isdir(gen_score_dir):
            continue

        real_score_dir = os.path.join(real_emb_dir, score_name)

        real_emb_lst = load_embeddings(real_score_dir)

        gen_emb_name = "perf_0.npy"
        gen_emb_path = os.path.join(gen_score_dir, gen_emb_name)
        gen_emb = np.load(gen_emb_path).squeeze()
        
        dist_lst = []
        for real_emb in real_emb_lst:
            dist = calculate_dtw_distance(real_emb.astype(np.double),
                                          gen_emb.astype(np.double))
            dist_lst.append(dist**0.5)
        
        dtw_dict[score_name] = dist_lst
    
    torch.save(dtw_dict, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_emb_dir', type=str, required=True, help='path to directory with real embeddings')
    parser.add_argument('--gen_emb_dir', type=str, required=True, help='path to directory with generated embeddings')
    parser.add_argument('--output_filename', type=str, required=True, help='filename of output dtw dictionary')

    args = parser.parse_args()

    main(args)
