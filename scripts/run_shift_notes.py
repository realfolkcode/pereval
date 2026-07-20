import os
import argparse
import torch
import numpy as np
import symusic
from symusic import Score, Note
from tqdm import tqdm


def main(args):
    evaluation_dir = args.evaluation_dir
    output_dir = args.output_dir
    shift = args.shift

    os.makedirs(output_dir, exist_ok=True)

    for i, score_name in enumerate(tqdm(os.listdir(evaluation_dir))):
        score_dir = os.path.join(evaluation_dir, score_name)
        if not os.path.isdir(score_dir):
            continue

        output_score_dir = os.path.join(output_dir, score_name)
        os.makedirs(output_score_dir, exist_ok=True)
        
        if os.path.isdir(score_dir):
            for perf_name in os.listdir(score_dir):
                if perf_name.endswith(".mid") or perf_name.endswith(".midi"):
                    perf_path = os.path.join(score_dir, perf_name)
                    mid = Score(perf_path)
                    mid.tracks[0].notes = mid.tracks[0].notes[shift:]
                    output_path = os.path.join(output_score_dir, perf_name)
                    mid.dump_midi(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_dir', type=str, required=True, help='path to evaluation dir')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--shift', type=int, required=True, help='number of notes to shift')

    args = parser.parse_args()

    main(args)
