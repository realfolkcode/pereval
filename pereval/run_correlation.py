import os
import argparse
import parangonar as pa
import partitura
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List

from pereval.alignment import NoteAligner
from pereval.correlation import calculate_correlation

import warnings
warnings.filterwarnings('ignore')


def get_dataset_correlation(real_dir: str,
                            gen_dir: str,
                            real_alignment_dir: str,
                            gen_alignment_dir: str,
                            root_score_dir: str) -> Dict[str, List]:
    note_aligner = NoteAligner()

    ioi_corr_dict = {}
    vel_corr_dict = {}
    duration_corr_dict = {}

    for score_name in tqdm(os.listdir(real_dir)):
        real_score_dir = os.path.join(real_dir, score_name)
        if not os.path.isdir(real_score_dir):
            continue
        score_path = os.path.join(root_score_dir, f"{score_name}.mid")
        score = partitura.load_score_midi(score_path).note_array()

        gen_score_dir = os.path.join(gen_dir, score_name)
        if not os.path.isdir(gen_score_dir):
            continue

        real_perf_lst = []
        real_score_perf_indices_lst = []
        gen_perf_lst = []
        gen_score_perf_indices_lst = []

        for real_perf_name in os.listdir(real_score_dir):
            if not real_perf_name.endswith(".mid"):
                continue
            real_perf_path = os.path.join(real_score_dir, real_perf_name)
            real_perf = partitura.load_performance_midi(real_perf_path)
            real_perf = real_perf.note_array()

            real_alignment_path = real_perf_name[:-4] + ".match"
            real_alignment_path = os.path.join(real_alignment_dir,
                                               score_name,
                                               real_alignment_path)

            real_score_perf_indices = note_aligner.load_score_perf_alignment(real_alignment_path,
                                                                             perf=real_perf,
                                                                             score=score)
            real_perf_lst.append(real_perf)
            real_score_perf_indices_lst.append(real_score_perf_indices)
            
        for gen_perf_name in os.listdir(gen_score_dir):
            if not gen_perf_name.endswith(".mid"):
                continue
            gen_perf_path = os.path.join(gen_score_dir, gen_perf_name)
            gen_perf = partitura.load_performance_midi(gen_perf_path)
            gen_perf = gen_perf.note_array()

            gen_alignment_path = gen_perf_name[:-4] + ".match"
            gen_alignment_path = os.path.join(gen_alignment_dir,
                                              score_name,
                                              gen_alignment_path)

            gen_score_perf_indices = note_aligner.load_score_perf_alignment(gen_alignment_path,
                                                                            perf=gen_perf,
                                                                            score=score)

            gen_perf_lst.append(gen_perf)
            gen_score_perf_indices_lst.append(gen_score_perf_indices)
        
        ioi_corr_dict[score_name] = torch.zeros((len(real_perf_lst),
                                                 len(gen_perf_lst)))
        vel_corr_dict[score_name] = torch.zeros((len(real_perf_lst),
                                                 len(gen_perf_lst)))
        duration_corr_dict[score_name] = torch.zeros((len(real_perf_lst),
                                                      len(gen_perf_lst)))
                
        for i in range(len(real_perf_lst)):
            real_perf = real_perf_lst[i]
            real_score_perf_indices = real_score_perf_indices_lst[i]
            for j in range(len(gen_perf_lst)):
                gen_perf = gen_perf_lst[j]
                gen_score_perf_indices = gen_score_perf_indices_lst[j]
                corr = calculate_correlation(score=score,
                                             real_perf=real_perf,
                                             gen_perf=gen_perf,
                                             real_score_perf_indices=real_score_perf_indices,
                                             gen_score_perf_indices=gen_score_perf_indices,
                                             note_aligner=note_aligner)
                ioi_corr_dict[score_name][i, j] = corr['ioi']
                vel_corr_dict[score_name][i, j] = corr['velocity']
                duration_corr_dict[score_name][i, j] = corr['duration']

    return {'ioi': ioi_corr_dict,
            'velocity': vel_corr_dict,
            'duration': duration_corr_dict}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, required=True, help='path to directory with real performances')
    parser.add_argument('--gen_dir', type=str, required=True, help='path to directory with generated performances')
    parser.add_argument('--real_alignment_dir', type=str, required=True, help='directory with alignments for real performances')
    parser.add_argument('--gen_alignment_dir', type=str, required=True, help='directory with alignments for generated performances')
    parser.add_argument('--score_dir', type=str, required=True, help='path to directory with scores')
    parser.add_argument('--output_filename', type=str, required=True, help='filename of output correlation')

    args = parser.parse_args()

    real_dir = args.real_dir
    gen_dir = args.gen_dir
    real_alignment_dir = args.real_alignment_dir
    gen_alignment_dir = args.gen_alignment_dir
    score_dir = args.score_dir
    output_filename = args.output_filename

    output_path = os.path.join(gen_dir, output_filename)

    corr_dict = get_dataset_correlation(real_dir=real_dir,
                                        gen_dir=gen_dir,
                                        real_alignment_dir=real_alignment_dir,
                                        gen_alignment_dir=gen_alignment_dir,
                                        root_score_dir=score_dir)
    torch.save(corr_dict, output_path)


if __name__ == "__main__":
    main()
