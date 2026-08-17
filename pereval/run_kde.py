import os
import argparse
import parangonar as pa
import partitura
import numpy as np
import torch
import skops.io as sio
from scipy.stats import gaussian_kde, entropy
from tqdm import tqdm
from typing import Dict, List

from pereval.alignment import NoteAligner
from pereval.correlation import aggregate

import warnings
warnings.filterwarnings('ignore')


def extract_features(perf,
                     score,
                     score_perf_indices,
                     note_aligner):
    vel = perf['velocity']
    duration = perf['duration_sec']

    onset_indices = note_aligner.get_onset_indices(score,
                                                   indices=score_perf_indices[0])

    onsets = perf['onset_sec'][score_perf_indices[1]]
    onsets = aggregate(onset_indices, onsets)
    ioi = onsets[1:] - onsets[:-1]

    return ioi, vel, duration


def get_dataset_kde(eval_dir,
                    root_score_dir) -> Dict:
    note_aligner = NoteAligner()

    ioi_lst = []
    vel_lst = []
    duration_lst = []

    for score_name in tqdm(os.listdir(eval_dir)):
        score_dir = os.path.join(eval_dir, score_name)
        if not os.path.isdir(score_dir):
            continue
        score_path = os.path.join(root_score_dir, f"{score_name}.mid")
        score = partitura.load_score_midi(score_path).note_array()

        for perf_name in os.listdir(score_dir):
            if not perf_name.endswith(".mid"):
                continue
            perf_path = os.path.join(score_dir, perf_name)
            perf = partitura.load_performance_midi(perf_path)
            perf = perf.note_array()

            alignment_path = perf_name[:-4] + ".match"
            alignment_path = os.path.join(score_dir, alignment_path)

            score_perf_indices = note_aligner.load_score_perf_alignment(alignment_path,
                                                                        perf=perf,
                                                                        score=score)
            
            ioi, vel, duration = extract_features(perf=perf,
                                                  score=score,
                                                  score_perf_indices=score_perf_indices,
                                                  note_aligner=note_aligner)
            ioi_lst.append(ioi)
            vel_lst.append(vel)
            duration_lst.append(duration)
    
    ioi = np.concatenate(ioi_lst)
    vel = np.concatenate(vel_lst)
    duration = np.concatenate(duration_lst)

    kde_ioi = gaussian_kde(ioi)
    kde_vel = gaussian_kde(vel)
    kde_duration = gaussian_kde(duration)

    return {'kde_ioi': kde_ioi,
            'kde_vel': kde_vel,
            'kde_duration': kde_duration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_dir', type=str, required=True, help='path to evaluation dir')
    parser.add_argument('--score_dir', type=str, required=True, help='path to directory with scores')
    parser.add_argument('--output_filename', type=str, required=True, help='filename of output KDE')

    args = parser.parse_args()

    eval_dir = args.evaluation_dir
    score_dir = args.score_dir
    output_filename = args.output_filename

    output_path = os.path.join(eval_dir, output_filename)

    kdes = get_dataset_kde(eval_dir=eval_dir,
                           root_score_dir=score_dir)
    
    sio.dump(kdes, output_path)


if __name__ == "__main__":
    main()
