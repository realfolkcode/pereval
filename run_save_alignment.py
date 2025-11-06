import os
import parangonar as pa
import partitura
import numpy as np
import argparse
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def main(args):
    score_dir = args.score_dir
    evaluation_dir = args.evaluation_dir

    matcher = pa.TheGlueNoteMatcher()

    for score_name in tqdm(os.listdir(evaluation_dir)):
        piece_dir = os.path.join(evaluation_dir, score_name)
        if not os.path.isdir(piece_dir):
            continue
        score_path = os.path.join(score_dir, f"{score_name}.mid")
        score = partitura.load_score_midi(score_path)

        for perf_name in os.listdir(piece_dir):
            if not perf_name.endswith(".mid"):
                continue

            match_name = perf_name[:-4] + ".match"
            match_path = os.path.join(piece_dir, match_name)
            if os.path.exists(match_path):
                continue

            perf_path = os.path.join(piece_dir, perf_name)
            perf = partitura.load_performance_midi(perf_path)

            alignment = matcher(score.note_array(),
                                perf.note_array())
            
            partitura.save_match(alignment=alignment,
                                 performance_data=perf,
                                 score_data=score,
                                 out=match_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True, help='path to dir with scores')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='path to evaluation dir')

    args = parser.parse_args()

    main(args)
